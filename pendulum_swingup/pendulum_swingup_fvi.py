from sympy import is_convex
from utils import *
from polynomial_integration_fvi import pendulum_setup
from pydrake.all import (MakeVectorVariable, Solve, SolverOptions, CommonSolverOption, Polynomial)
from scipy.integrate import quad
from pydrake.solvers import mathematicalprogram as mp

import matplotlib.pyplot as plt
from matplotlib import cm


def iterative_hjb_sampling_regression(poly_deg, params_dict, rho=0):
    # Iterative convex least squares regression using sampling instead of integration
    nz = params_dict["nz"]
    coeff_shape = np.ones(nz, dtype=int) * (poly_deg + 1)
    old_coeff = np.zeros(coeff_shape)
    f = params_dict["f"]
    l = params_dict["l"]
    f2 = params_dict["f2"]
    x2z = params_dict["x2z"]
    z_max = np.array([1, 1, 2*np.pi])
    z_min = -z_max
    def poly_func(x, n): return x**n  

    n_mesh = 51
    mesh_pts = np.linspace(params_dict["x_min"], params_dict["x_max"], n_mesh)
    mesh_pts[int(np.floor(n_mesh/2))] = 0

    z = MakeVectorVariable(nz, "x")
    for i in range(100):
        print("Iter: ", i)
        dJdz = calc_dJdz(z, old_coeff, poly_func)
        prog = MathematicalProgram()
        prog.AddIndeterminates(z)
        J_coeff_var = prog.NewContinuousVariables(np.product(coeff_shape),
                                                  "J")
        J_coeff = np.array(J_coeff_var).reshape(coeff_shape)

        u_opt = calc_u_opt(dJdz, f2, params_dict["Rinv"])

        dJdz = calc_dJdz(z, J_coeff, poly_func)
        J = calc_value_function(z, J_coeff, poly_func)

        RHS = (l(z, u_opt) + dJdz.dot(f(z, u_opt)) - rho * J)**2
        for i in range(n_mesh):
            theta = mesh_pts[i, 0]
            for j in range(n_mesh):
                thetadot = mesh_pts[j, 1]
                x = np.array([theta, thetadot])
                z_val = x2z(x)
                prog.AddLinearConstraint(
                    calc_value_function(z_val, J_coeff, poly_func) >= 0)
                prog.AddQuadraticCost(RHS.EvaluatePartial(dict(zip(z,z_val))), is_convex=True)
            
        # J(z0) = 0
        J0 = calc_value_function(params_dict["z0"], J_coeff, poly_func)
        prog.AddLinearConstraint(J0 == 0)
        # options = SolverOptions()
        # options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        # prog.SetSolverOptions(options)
        result = Solve(prog)

        J_star = result.GetSolution(J_coeff_var).reshape(coeff_shape)

        assert result.is_success()
        print(result.is_success())
        # print(result.get_solver_id().name())

        if np.allclose(J_star, old_coeff, atol=1e-5):
            print("Error: ", result.get_optimal_cost())
            plot_value_function(J_star, params_dict, poly_func, poly_deg)
            break
        else:
            print("Diff: ", np.linalg.norm(J_star-old_coeff))
            old_coeff = J_star

    plot_value_function(J_star, params_dict, poly_func, poly_deg, file_name="iterative_hjb_sampling_regression")

    return J_star

def iterative_hjb_integration_regression(poly_deg, params_dict, rho=0):
    # Iterative convex least squares regression using integration
    nz = params_dict["nz"]
    coeff_shape = np.ones(nz, dtype=int) * (poly_deg + 1)
    old_coeff = np.zeros(coeff_shape)
    f = params_dict["f"]
    l = params_dict["l"]
    f2 = params_dict["f2"]
    x2z = params_dict["x2z"]
    z_max = np.array([1, 1, 2*np.pi])
    z_min = -z_max
    def poly_func(x, n): return x**n  

    n_mesh = 51
    mesh_pts = np.linspace(params_dict["x_min"], params_dict["x_max"], n_mesh)
    mesh_pts[int(np.floor(n_mesh/2))] = 0

    z = MakeVectorVariable(nz, "x")
    for i in range(100):
        print("Iter: ", i)
        dJdz = calc_dJdz(z, old_coeff, poly_func)
        prog = MathematicalProgram()
        prog.AddIndeterminates(z)
        J_coeff_var = prog.NewContinuousVariables(np.product(coeff_shape),
                                                  "J")
        J_coeff = np.array(J_coeff_var).reshape(coeff_shape)

        u_opt = calc_u_opt(dJdz, f2, params_dict["Rinv"])

        dJdz = calc_dJdz(z, J_coeff, poly_func)
        J = calc_value_function(z, J_coeff, poly_func)

        diff_int = Polynomial((l(z, u_opt) + dJdz.dot(f(z, u_opt)) - rho * J)**2, z)
        for i in range(nz):
            diff_int = diff_int.Integrate(z[i], z_min[i], z_max[i])
        prog.AddQuadraticCost(diff_int.ToExpression(), is_convex=True)
        # diff_int = diff_int.Integrate(z[-1], z_min[-1], z_max[-1])
        # for monomial,coeff in diff_int.monomial_to_coefficient_map().items(): 
        #     s_deg = monomial.degree(z[0]) 
        #     c_deg = monomial.degree(z[1])
        #     monomial_int = quad(lambda x: np.sin(x)**s_deg * np.cos(x)**c_deg, 0, 2*np.pi)[0]
        #     prog.AddQuadraticCost(monomial_int * coeff, is_convex=True)

        for i in range(n_mesh):
            theta = mesh_pts[i, 0]
            for j in range(n_mesh):
                thetadot = mesh_pts[j, 1]
                x = np.array([theta, thetadot])
                z_val = x2z(x)
                prog.AddLinearConstraint(
                    calc_value_function(z_val, J_coeff, poly_func) >= 0)
            
        # J(z0) = 0
        J0 = calc_value_function(params_dict["z0"], J_coeff, poly_func)
        prog.AddLinearConstraint(J0 == 0)
        # options = SolverOptions()
        # options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        # prog.SetSolverOptions(options)
        result = Solve(prog)

        J_star = result.GetSolution(J_coeff_var).reshape(coeff_shape)

        assert result.is_success()
        print(result.is_success())
        # print(result.get_solver_id().name())

        if np.allclose(J_star, old_coeff, atol=1e-5):
            print("Error: ", result.get_optimal_cost())
            plot_value_function(J_star, params_dict, poly_func, poly_deg)
            break
        else:
            print("Diff: ", np.linalg.norm(J_star-old_coeff))
            old_coeff = J_star

    plot_value_function(J_star, params_dict, poly_func, poly_deg)

    return J_star

def plot_value_function(coeff, params_dict, poly_func, deg, file_name="pendulum_swingup_policy"):
    x2z = params_dict["x2z"]
    x_min = params_dict["x_min"]
    x_max = params_dict["x_max"]
    z = MakeVectorVariable(params_dict["nz"], "z")

    X1, X2 = np.meshgrid(np.linspace(x_min[0], x_max[0], 51),
                        np.linspace(x_min[1], x_max[1], 51))
    X = np.vstack((X1.flatten(), X2.flatten()))
    Z = x2z(X)
    J = np.zeros(Z.shape[1])
    U = np.zeros(Z.shape[1])
    dJdz = calc_dJdz(z, coeff, poly_func)
    J0 = calc_value_function(params_dict["z0"], coeff, poly_func)
    u_opt_expr = calc_u_opt(dJdz, params_dict["f2"], params_dict["Rinv"]) 
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
    plt.savefig("figures/pendulum/{}}_{}.png".format(file_name, deg))

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
    plt.savefig("figures/pendulum/{}_{}.png".format(file_name, deg))

if __name__ == '__main__':
    poly_deg = 4
    params_dict = pendulum_setup()
    iterative_hjb_sampling_regression(poly_deg, params_dict)
    