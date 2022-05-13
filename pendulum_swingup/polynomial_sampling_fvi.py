import numpy as np
from scipy.special import comb
from pydrake.examples.pendulum import (PendulumParams)
from pydrake.all import (MathematicalProgram, Solve, SolverOptions, CommonSolverOption, MosekSolver)

from matplotlib import cm
import matplotlib.pyplot as plt

BERNSTEIN = "bernstein"
CHEBYSHEV = "chebyshev"
LEGENDRE = "legendre"
MONOMIAL = "monomial"
HERMITE = "hermite"
DRAKE = "drake"
LSTSQ = "lstsq"
BARYCENTRIC = "barycentric"

def pendulum_setup(poly_type=CHEBYSHEV):
    nz = 3

    # Map from original state to augmented state.
    # Uses sympy to be able to do symbolic integration later on.
    if poly_type == BERNSTEIN:
        x2z = lambda x : np.array([(np.sin(x[0]) + 1)/2, (np.cos(x[0])+1)/2, (x[1]+2*np.pi)/(4*np.pi)])
    elif poly_type == MONOMIAL or poly_type == HERMITE:
        x2z = lambda x : np.array([np.sin(x[0]), np.cos(x[0]), x[1]])
    else:
        x2z = lambda x : np.array([np.sin(x[0]), np.cos(x[0]), x[1]/(2*np.pi)])

    # System dynamics in augmented state (z).
    params = PendulumParams()
    inertia = params.mass() * params.length() ** 2
    tau_g = params.mass() * params.gravity() * params.length()
    def f(z, u):
        return np.array([
            z[1] * z[2],
            - z[0] * z[2],
            (tau_g * z[0] + u[0] - params.damping() * z[2]) / inertia
        ])
    def f_x(x, u):
        return np.array([
            x[1],
            (tau_g * np.sin(x[0]) + u[0] - params.damping() * x[1]) / inertia
        ])

    # State limits (region of state space where we approximate the value function).
    x_max = np.array([np.pi, 2*np.pi])
    x_min = - x_max

    # Equilibrium point in both the system coordinates.
    x0 = np.array([0, 0])
    z0 = x2z(x0)
        
    # Quadratic running cost in augmented state.
    Q = np.diag([1, 1, 1]) * 2
    Qx = np.diag([1, 1]) * 2
    R = np.diag([1]) 
    def l(z, u):
        return (z - z0).dot(Q).dot(z - z0) + u.dot(R).dot(u)
    def l_x(x, u):
        return (x - x0).dot(Qx).dot(x - x0) + u.dot(R).dot(u)

    Rinv = np.linalg.inv(R)
    f2 = np.array([[0], [0], [1 / inertia]])
    params_dict = {"x_min": x_min, "x_max": x_max, "x2z":x2z, "f":f, "l":l,
                   "nz": nz, "f2": f2, "Rinv": Rinv, "z0": z0, "f_x": f_x, "l_x": l_x}
    return params_dict


def monomial(x, n, d):
    return x**n


def bernstein_polynomial(t, i, n):
    c = comb(n, i)
    return c * t**i * (1-t)**(n-i)


def chebyshev_polynomial(x, n, d):
    v = 0
    for k in range(int(np.floor(n/2)) + 1):
        v += (-1)**k * comb(n-k, k)*(2*x)**(n-2*k)
    return v


def legendre_polynomial(x, n, d):
    v = 0
    for k in range(n + 1):
        v += comb(n, k) * comb(n+k, k) * ((x-1)/2)**k
    return v


def hermite_polynomial(x, n, d):
    v = 0
    for k in range(int(np.floor(n/2)) + 1):
        v += (-1)**k * (2*x)**(n-2*k) /(np.math.factorial(k) * np.math.factorial(n - 2*k))
    v *= np.math.factorial(n)
    return v


def fitted_value_iteration_drake(poly_deg, params_dict, poly_type=CHEBYSHEV,dt=0.1, gamma=1, old_target=True):
    n_mesh = 51
    mesh_pts = np.linspace(params_dict["x_min"], params_dict["x_max"], n_mesh)
    mesh_pts[int(np.floor(n_mesh/2))] = 0
    coeff_shape = np.ones(params_dict["nz"], dtype=int) * (poly_deg + 1)
    old_coeff = np.zeros(coeff_shape)
    x2z = params_dict["x2z"]
    f = params_dict["f"]
    f_x = params_dict["f_x"]
    l = params_dict["l"]

    if poly_type == CHEBYSHEV:
        poly_func = lambda t, i, n: chebyshev_polynomial(t, i, n)
    elif poly_type == LEGENDRE:
        poly_func = lambda t, i, n: legendre_polynomial(t, i, n)
    elif poly_type == BERNSTEIN:
        poly_func = lambda t, i, n: bernstein_polynomial(t, i, n)
    elif poly_type == HERMITE:
        poly_func = lambda t, i, n: hermite_polynomial(t, i, n)
    elif poly_type == MONOMIAL:
        poly_func = lambda t, i, n: monomial(t, i, n)

    for iter in range(100):
        print("Iter: ", iter)
        dJdz_expr, z_var = calc_dJdz(old_coeff, poly_func, params_dict)
        prog = MathematicalProgram()
        J_coeff_var = prog.NewContinuousVariables(np.product(coeff_shape),
                                                        "J")
        J_coeff = np.array(J_coeff_var).reshape(coeff_shape)
        for i in range(n_mesh):
            theta = mesh_pts[i, 0]
            for j in range(n_mesh):
                thetadot = mesh_pts[j, 1]
                x = np.array([theta, thetadot])
                z = x2z(x)
                u_opt = calc_u_opt(dJdz_expr, z_var, z, params_dict)
                x_next = f_x(x, u_opt) * dt + x
                z_next = x2z(x_next)
                # z_next = f(z, u_opt) * dt + z
                if old_target:
                    J_target = l(z, u_opt) * dt + gamma * calc_value_function(z_next,
                                                                  old_coeff, poly_func) 
                else:
                    J_target = l(z, u_opt) * dt + gamma * calc_value_function(z_next,
                                                                    J_coeff, poly_func)                                 
                J = calc_value_function(z, J_coeff, poly_func)

                prog.AddLinearConstraint(J >= 0)
                prog.AddQuadraticCost((J-J_target)*(J-J_target))

        # J(z0) = 0
        J0 = calc_value_function(params_dict["z0"], J_coeff, poly_func)
        prog.AddLinearConstraint(J0 == 0)  
        options = SolverOptions()
        options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        prog.SetSolverOptions(options)
        result = Solve(prog)

        coeff = result.GetSolution(J_coeff_var).reshape(coeff_shape)
        print("Diff: ", np.linalg.norm(coeff-old_coeff))
        if np.allclose(coeff, old_coeff, atol=1e-3):
            plot_value_function(coeff, params_dict, poly_func, deg, dt, poly_type, DRAKE)
            return coeff

        if result.is_success():
            old_coeff = coeff
        else:
            print("Optimizer fails")
            
    return coeff


def fitted_value_iteration_lstsq(poly_deg, params_dict, poly_type=CHEBYSHEV,dt=0.1, gamma=1, old_target=True):
    n_mesh = 51
    mesh_pts = np.linspace(params_dict["x_min"], params_dict["x_max"], n_mesh)
    mesh_pts[int(np.floor(n_mesh/2))] = 0
    coeff_shape = np.ones(params_dict["nz"], dtype=int) * (poly_deg + 1)
    old_coeff = np.load("pendulum_swingup/data/{}/{}/J_{}_{}.npy".format(DRAKE, poly_type, deg, dt)) #np.zeros(coeff_shape)
    x2z = params_dict["x2z"]
    f = params_dict["f"]
    f_x = params_dict["f_x"]
    l = params_dict["l"]
    l_x = params_dict["l_x"]

    if poly_type == CHEBYSHEV:
        poly_func = lambda t, i, n: chebyshev_polynomial(t, i, n)
    elif poly_type == LEGENDRE:
        poly_func = lambda t, i, n: legendre_polynomial(t, i, n)
    elif poly_type == BERNSTEIN:
        poly_func = lambda t, i, n: bernstein_polynomial(t, i, n)
    elif poly_type == HERMITE:
        poly_func = lambda t, i, n: hermite_polynomial(t, i, n)
    elif poly_type == MONOMIAL:
        poly_func = lambda t, i, n: monomial(t, i, n)
    Z = []

    for iter in range(500):
        print("Iter: ", iter)
        dJdz_expr, z_var = calc_dJdz(old_coeff, poly_func, params_dict)
        J_target = []
        l_target = []
        Z_next = []
        for i in range(n_mesh):     
            theta = mesh_pts[i, 0]  
            for j in range(n_mesh):
                thetadot = mesh_pts[j, 1]
                x = np.array([theta, thetadot])
                z = x2z(x)
                if iter == 0:
                    basis = calc_basis(z, coeff_shape, poly_func)
                    Z.append(basis)

                u_opt = calc_u_opt(dJdz_expr, z_var, z, params_dict)
                x_next = f_x(x, u_opt) * dt + x
                z_next = x2z(x_next)
                # z_next0 = f(z, u_opt) * dt + z
                if poly_type != MONOMIAL and poly_type != HERMITE:
                    z_next = np.clip(z_next, -1, 1)

                if old_target:
                    J_target.append(l(z, u_opt) * dt + gamma * old_coeff.flatten().dot(calc_basis(z_next, coeff_shape, poly_func)))
                else:
                    l_target.append(l(z, u_opt) * dt)
                    Z_next.append(calc_basis(z_next, coeff_shape, poly_func))                             
                
        if iter == 0:
            Z = np.array(Z)
        if old_target:
            J_target = np.array(J_target)
            coeff = np.linalg.lstsq(Z, J_target)[0].reshape(coeff_shape)
        else:
            Z_next = np.array(Z_next)
            coeff = np.linalg.lstsq((Z-gamma *Z_next), l_target)[0].reshape(coeff_shape)
        
        
        if np.allclose(coeff, old_coeff):
            plot_value_function(coeff, params_dict, poly_func, deg, dt, poly_type)
            return coeff
        else:
            print("Diff: ", np.linalg.norm(coeff-old_coeff))
            old_coeff = coeff

    return coeff


def fit_barycentric_fvi(poly_deg, params_dict):
    n_mesh = 51
    mesh_pts = np.linspace(params_dict["x_min"], params_dict["x_max"], n_mesh)
    mesh_pts[int(np.floor(n_mesh/2))] = 0
    coeff_shape = np.ones(params_dict["nz"], dtype=int) * (poly_deg + 1)
    x2z = params_dict["x2z"]

    if poly_type == CHEBYSHEV:
        poly_func = lambda t, i, n: chebyshev_polynomial(t, i, n)
    elif poly_type == LEGENDRE:
        poly_func = lambda t, i, n: legendre_polynomial(t, i, n)
    elif poly_type == BERNSTEIN:
        poly_func = lambda t, i, n: bernstein_polynomial(t, i, n)
    elif poly_type == HERMITE:
        poly_func = lambda t, i, n: hermite_polynomial(t, i, n)
    elif poly_type == MONOMIAL:
        poly_func = lambda t, i, n: monomial(t, i, n)
    Z = []
    Jd = np.load("pendulum_swingup/data/J.npy")

    for i in range(n_mesh):     
        theta = mesh_pts[i, 0]  
        for j in range(n_mesh):
            thetadot = mesh_pts[j, 1]
            x = np.array([theta, thetadot])
            z = x2z(x)
            basis = calc_basis(z, coeff_shape, poly_func)
            Z.append(basis)
    Z = np.array(Z)
    J = np.linalg.lstsq(Z, Jd.flatten())[0].reshape(coeff_shape)
    plot_value_function(J, params_dict, poly_func, deg, dt, poly_type, method=method)
    np.save("pendulum_swingup/data/{}/{}/J_{}_{}.npy".format(method, poly_type, deg, dt), J)


def plot_value_function(coeff, params_dict, poly_func, deg, dt, poly_type=MONOMIAL, method=LSTSQ):
    x2z = params_dict["x2z"]
    x_min = params_dict["x_min"]
    x_max = params_dict["x_max"]

    X1, X2 = np.meshgrid(np.linspace(x_min[0], x_max[0], 51),
                        np.linspace(x_min[1], x_max[1], 51))
    X = np.vstack((X1.flatten(), X2.flatten()))
    Z = x2z(X)
    J = np.zeros(Z.shape[1])
    U = np.zeros(Z.shape[1])
    dJdz_expr, z_var = calc_dJdz(coeff, poly_func, params_dict)
    J0 = calc_value_function(params_dict["z0"], coeff, poly_func)
    for i in range(Z.shape[1]):
        J[i] = calc_value_function(Z[:,i], coeff, poly_func)
        U[i] = calc_u_opt(dJdz_expr, z_var, Z[:,i], params_dict) 
    J = J.reshape(X1.shape) - J0
    U = U.reshape(X1.shape)
    
    fig = plt.figure(figsize=(9, 4))
    ax = fig.subplots()
    ax.set_xlabel("q")
    ax.set_ylabel("qdot")
    ax.set_title("Cost-to-Go")
    im = ax.imshow(J,
            cmap=cm.jet, aspect='auto',
            extent=(x_min[0], x_max[0], x_min[1], x_max[1]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.savefig("figures/fvi/{}/fvi_pendulum_swingup_{}_{}_{}.png".format(method, deg, dt, poly_type))

    fig = plt.figure(figsize=(9, 4))
    ax = fig.subplots()
    ax.set_xlabel("q")
    ax.set_ylabel("qdot")
    ax.set_title("Optimal Policy")
    im = ax.imshow(U,
            cmap=cm.jet, aspect='auto',
            extent=(x_min[0], x_max[0], x_min[1], x_max[1]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.savefig("figures/fvi/{}/fvi_pendulum_swingup_policy_{}_{}_{}.png".format(method, deg, dt, poly_type))


def calc_dJdz(coeff, poly_func, params_dict):
    nz = params_dict["nz"]
    prog = MathematicalProgram()
    z = prog.NewIndeterminates(nz, 'z')
    J = calc_value_function(z, coeff, poly_func)
    dJdz_expr = J.Jacobian(z)
    return dJdz_expr, z


def calc_u_opt(dJdz_expr, z_var, z_value, params_dict):
    f2 = params_dict["f2"]
    dJdz = np.array([k.Evaluate(dict(zip(z_var, z_value))) for k in dJdz_expr])
    u_star = - .5 * params_dict["Rinv"].dot(f2.T).dot(dJdz.T)
    return u_star


def calc_basis(z, coeff_shape, poly_func):
    Z = np.zeros(coeff_shape)
    deg = coeff_shape - 1
    it = np.nditer(Z, flags=['multi_index'])
    for x in it:
        idx = it.multi_index
        b = 1
        for dim in range(len(idx)):
            b *= poly_func(z[dim], idx[dim], deg[dim])
        Z[idx] = b
    return Z.flatten()


def calc_value_function(x, K, poly_func):
    assert len(x) == len(K.shape)
    it = np.nditer(K, flags=['multi_index', 'refs_ok'])
    p = 0
    for k in it:
        b = np.copy(k)
        for dim, idx in enumerate(it.multi_index):
            b = b*poly_func(x[dim], idx, K.shape[dim]-1)
        p += b
    return p


if __name__ == '__main__':
    method = BARYCENTRIC
    poly_type = CHEBYSHEV
    params_dict = pendulum_setup(poly_type)
    deg = 6
    dt = 0.01
    fit_barycentric_fvi(deg, params_dict)
    if method == LSTSQ:
        J = fitted_value_iteration_lstsq(deg, params_dict, poly_type, dt=dt, gamma=.9, old_target=False)
    elif method == DRAKE:
        J = fitted_value_iteration_drake(deg, params_dict, poly_type, dt=dt, gamma=.999, old_target=False)
        
    np.save("pendulum_swingup/data/{}/{}/J_{}_{}.npy".format(method, poly_type, deg, dt), J)