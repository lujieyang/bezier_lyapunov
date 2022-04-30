import numpy as np
from pydrake.examples.pendulum import (PendulumParams)
from pydrake.all import (MathematicalProgram, Solve, SolverOptions, CommonSolverOption, Polynomial)

from matplotlib import cm
import matplotlib.pyplot as plt

def pendulum_setup():
    nz = 3

    # Map from original state to augmented state.
    # Uses sympy to be able to do symbolic integration later on.
    x2z = lambda x : np.array([np.sin(x[0]), np.cos(x[0]), x[1]])


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


def monomial(x, n):
    return x**n


def calc_basis(z, coeff_shape, poly_func):
    Z = np.zeros(coeff_shape)
    it = np.nditer(Z, flags=['multi_index'])
    for x in it:
        idx = it.multi_index
        b = 1
        for dim in range(len(idx)):
            b *= poly_func(z[dim], idx[dim])
        Z[idx] = b
    return Z.flatten()


def calc_dJdz(coeff, poly_func, nz):
    prog = MathematicalProgram()
    z = prog.NewIndeterminates(nz, 'z')
    J = calc_value_function(z, coeff, poly_func)
    dJdz_expr = J.Jacobian(z)
    return dJdz_expr, z


def calc_value_function(x, J, poly_func):
    assert len(x) == len(J.shape)
    it = np.nditer(J, flags=['multi_index', 'refs_ok'])
    p = 0
    for k in it:
        b = np.copy(k)
        for dim, idx in enumerate(it.multi_index):
            b = b*poly_func(x[dim], idx)
        p += b
    return p


def calc_u_opt(dJdz, f2):
    u_star = - .5 * params_dict["Rinv"].dot(f2.T).dot(dJdz.T)
    return u_star


def fitted_value_iteration(poly_deg, params_dict, dt, gamma=1):
    nz = params_dict["nz"]
    coeff_shape = np.ones(nz, dtype=int) * (poly_deg + 1)
    old_coeff = np.zeros(coeff_shape)
    x2z = params_dict["x2z"]
    f = params_dict["f"]
    f_x = params_dict["f_x"]
    l = params_dict["l"]
    f2 = params_dict["f2"]
    z_max = np.array([1, 1, 2*np.pi])
    z_min = -z_max
    poly_func = lambda x, n: monomial(x, n)

    for iter in range(100):
        print("Iter: ", iter)
        dJdz, z = calc_dJdz(old_coeff, poly_func, nz)
        prog = MathematicalProgram()
        J_coeff_var = prog.NewContinuousVariables(np.product(coeff_shape),
                                                        "J")
        J_coeff = np.array(J_coeff_var).reshape(coeff_shape)
        # dJdz, z = calc_dJdz(J_coeff, poly_func, nz)

        J = calc_value_function(z, J_coeff, poly_func)
        u_opt = calc_u_opt(dJdz, f2)
        z_next = f(z, u_opt) * dt + z
        J_target = l(z, u_opt) * dt + gamma * calc_value_function(z_next,
                                                                  J_coeff, poly_func) 

        
        diff_int = Polynomial((J - J_target)**2)
        for i in range(nz):
            diff_int = diff_int.Integrate(z[i], z_min[i], z_max[i])
        prog.AddCost(diff_int.ToExpression())

        # J(z0) = 0
        J0 = calc_value_function(params_dict["z0"], J_coeff, poly_func)
        prog.AddLinearConstraint(J0 == 0)  
        options = SolverOptions()
        options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        prog.SetSolverOptions(options)
        result = Solve(prog)

        coeff = result.GetSolution(J_coeff_var).reshape(coeff_shape)
        print("Diff: ", np.linalg.norm(coeff-old_coeff))

        if np.allclose(coeff, old_coeff):
            plot_value_function(coeff, params_dict, poly_func, poly_deg, dt)
            return coeff

        if result.is_success():
            old_coeff = coeff
        else:
            print("Optimizer fails")
            
    return coeff

def plot_value_function(coeff, params_dict, poly_func, deg, dt):
    x2z = params_dict["x2z"]
    x_min = params_dict["x_min"]
    x_max = params_dict["x_max"]

    X1, X2 = np.meshgrid(np.linspace(x_min[0], x_max[0], 51),
                        np.linspace(x_min[1], x_max[1], 51))
    X = np.vstack((X1.flatten(), X2.flatten()))
    Z = x2z(X)
    J = np.zeros(Z.shape[1])
    U = np.zeros(Z.shape[1])
    dJdz, z = calc_dJdz(coeff, poly_func, params_dict["nz"])
    J0 = calc_value_function(params_dict["z0"], coeff, poly_func)
    u_opt_expr = calc_u_opt(dJdz, params_dict["f2"]) 
    for i in range(Z.shape[1]):
        J[i] = calc_value_function(Z[:,i], coeff, poly_func)
        U[i] = u_opt_expr[0].Evaluate(dict(zip(z, Z[:, i])))
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
    plt.savefig("figures/fvi/fvi_pendulum_swingup_{}_{}.png".format(deg, dt))

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
    plt.savefig("figures/fvi/fvi_pendulum_swingup_policy_{}_{}.png".format(deg, dt))


if __name__ == '__main__':
    poly_deg = 2
    dt = 0.01
    params_dict = pendulum_setup()
    J = fitted_value_iteration(poly_deg, params_dict, dt)

    np.save("pendulum_swingup/J_{}_{}.npy".format(poly_deg, dt), J)