import numpy as np
from pydrake.examples.pendulum import (PendulumParams)
from pydrake.all import (MathematicalProgram, Solve, SolverOptions, CommonSolverOption)
from matplotlib import cm
import matplotlib.pyplot as plt

BERNSTEIN = "bernstein"
CHEBYSHEV = "chebyshev"

def pendulum_setup(poly_type=CHEBYSHEV):
    nz = 3

    # Map from original state to augmented state.
    # Uses sympy to be able to do symbolic integration later on.
    if poly_type == BERNSTEIN:
        x2z = lambda x : np.array([(np.sin(x[0]) + 1)/2, (np.cos(x[0])+1)/2, (x[1]+2*np.pi)/(4*np.pi)])
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

    # State limits (region of state space where we approximate the value function).
    x_max = np.array([np.pi, 2*np.pi])
    x_min = - x_max

    # Equilibrium point in both the system coordinates.
    x0 = np.array([0, 0])
    z0 = x2z(x0)
        
    # Quadratic running cost in augmented state.
    Q = np.diag([1, 1, 1])
    R = np.diag([1])
    def l(z, u):
        return (z - z0).dot(Q).dot(z - z0) + u.dot(R).dot(u)

    Rinv = np.linalg.inv(R)
    f2 = np.array([[0], [0], [1 / inertia]])
    params_dict = {"x_min": x_min, "x_max": x_max, "x2z":x2z, "f":f, "l":l,
                   "nz": nz, "f2": f2, "Rinv": Rinv}
    return params_dict


def bernstein_polynomial(t, i, n):
    c = np.math.factorial(n) / (np.math.factorial(i) * np.math.factorial(n - i))
    return c * t**i * (1-t)**(n-i)


def fitted_value_iteration_drake(poly_deg, params_dict, dt=0.1):
    n_mesh = 21
    mesh_pts = np.linspace(params_dict["x_min"], params_dict["x_max"], n_mesh)
    coeff_shape = np.ones(params_dict["nz"], dtype=int) * (poly_deg + 1)
    old_coeff = np.zeros(coeff_shape)
    x2z = params_dict["x2z"]
    f = params_dict["f"]
    l = params_dict["l"]

    poly_func = lambda t, i, n: bernstein_polynomial(t, i, n)

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
                z = x2z([theta, thetadot])
                u_opt = calc_u_opt(dJdz_expr, z_var, z, params_dict)
                z_next = f(z, u_opt) * dt + z

                J_target = l(z, u_opt) * dt + calc_value_function(z_next,
                                                                  old_coeff, poly_func)                                
                J = calc_value_function(z, J_coeff, poly_func)
                prog.AddQuadraticCost((J-J_target)*(J-J_target))
        
        options = SolverOptions()
        options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        prog.SetSolverOptions(options)
        result = Solve(prog)

        coeff = result.GetSolution(J_coeff_var).reshape(coeff_shape)

        if np.allclose(coeff, old_coeff):
            plot_value_function(coeff, mesh_pts, params_dict, poly_func)
            return coeff

        if result.is_success():
            old_coeff = coeff
        else:
            print("Optimizer fails")


def fitted_value_iteration(poly_deg, params_dict, dt=0.1, gamma=1):
    n_mesh = 51
    mesh_pts = np.linspace(params_dict["x_min"], params_dict["x_max"], n_mesh)
    coeff_shape = np.ones(params_dict["nz"], dtype=int) * (poly_deg + 1)
    old_coeff = np.zeros(coeff_shape)
    x2z = params_dict["x2z"]
    f = params_dict["f"]
    l = params_dict["l"]

    poly_func = lambda t, i, n: bernstein_polynomial(t, i, n)
    Z = []

    for iter in range(100):
        print("Iter: ", iter)
        dJdz_expr, z_var = calc_dJdz(old_coeff, poly_func, params_dict)
        J_target = []
        for i in range(n_mesh):     
            theta = mesh_pts[i, 0]  
            for j in range(n_mesh):
                thetadot = mesh_pts[j, 1]
                z = x2z([theta, thetadot])
                if iter == 0:
                    basis = calc_bernstein_basis(z, coeff_shape)
                    Z.append(basis)

                u_opt = calc_u_opt(dJdz_expr, z_var, z, params_dict)
                z_next = f(z, u_opt) * dt + z

                J_target.append(l(z, u_opt) * dt + gamma * calc_value_function(z_next,
                                                                  old_coeff, poly_func))                                
                
        if iter == 0:
            Z = np.array(Z)
        J_target = np.array(J_target)
        coeff = np.linalg.lstsq(Z, J_target)[0].reshape(coeff_shape)
        
        if np.allclose(coeff, old_coeff):
            plot_value_function(coeff, mesh_pts, params_dict, poly_func)
            return coeff
        else:
            print("Error: ", np.linalg.norm(coeff-old_coeff))
            old_coeff = coeff

    return coeff


def plot_value_function(coeff, mesh_pts, params_dict, poly_func):
    # [X, Y] = np.meshgrid(mesh_pts[:, 0], mesh_pts[:, 1])
    x2z = params_dict["x2z"]
    x_min = params_dict["x_min"]
    x_max = params_dict["x_max"]


    n_mesh = mesh_pts.shape[0]
    J = np.zeros([n_mesh, n_mesh])
    for i in range(n_mesh):
        theta = mesh_pts[i, 0]
        for j in range(n_mesh):
            thetadot = mesh_pts[j, 1]
            z = x2z([theta, thetadot])
            J[i, j] = calc_value_function(z, coeff, poly_func)

    fig = plt.figure(figsize=(9, 4))
    ax = fig.subplots()
    ax.set_xlabel("q")
    ax.set_ylabel("qdot")
    ax.set_title("Cost-to-Go")
    ax.imshow(J,
            cmap=cm.jet, aspect='auto',
            extent=(x_min[0], x_max[0], x_min[1], x_max[1]))
    ax.invert_yaxis()
    plt.savefig("fvi_pendulum_swingup.png")


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


def calc_bernstein_basis(z, coeff_shape):
    Z = np.zeros(coeff_shape)
    deg = coeff_shape - 1
    it = np.nditer(Z, flags=['multi_index'])
    for x in it:
        idx = it.multi_index
        b = 1
        for dim in range(len(idx)):
            b *= bernstein_polynomial(z[dim], idx[dim], deg[dim])
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
    params_dict = pendulum_setup(BERNSTEIN)
    deg = 2
    fitted_value_iteration(deg, params_dict, dt=0.1, gamma=1)