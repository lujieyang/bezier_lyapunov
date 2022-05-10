import numpy as np
import pydrake
from pydrake.solvers import mathematicalprogram as mp
from pydrake.all import (MathematicalProgram, Solve, SolverOptions,
                         CommonSolverOption, Polynomial, GurobiSolver, Variables, Variable)
from utils import *
import pydrake.symbolic as sym

import matplotlib.pyplot as plt


def cubic_setup():
    nz = 1
    z0 = 0

    def f(x, u):
        return np.array([x - 4*x**3 + u])

    f2 = np.array([1])

    Q = 1
    R = 1

    def l(x, u):
        return Q*x**2 + R*u**2

    return {"nz": nz, "f": f, "f2": f2, "l": l, "Rinv": np.diag([1/R]), "z0": z0}


def nonconvex_hjb_regression(poly_deg, params_dict):
    # Need positivity for J, nonconvex least square regression
    nz = params_dict["nz"]
    coeff_shape = np.ones(nz, dtype=int) * (poly_deg + 1)
    f = params_dict["f"]
    l = params_dict["l"]
    f2 = params_dict["f2"]
    z_max = np.array([1])
    z_min = -z_max
    def poly_func(x, n): return x**n  # monomial(x, n)

    # dJdz, z = calc_dJdz(old_coeff, poly_func, nz)
    prog = MathematicalProgram()
    J_coeff_var = prog.NewContinuousVariables(np.product(coeff_shape),
                                              "J")
    J_coeff = np.array(J_coeff_var).reshape(coeff_shape)

    z = prog.NewIndeterminates(nz, "z")
    dJdz = calc_dJdz(z, J_coeff, poly_func)

    u_opt = calc_u_opt(dJdz, f2, params_dict["Rinv"])
    diff_int = Polynomial((l(z, u_opt) + dJdz.dot(f(z, u_opt))).squeeze()**2)
    for i in range(nz):
        diff_int = diff_int.Integrate(z[i], z_min[i], z_max[i])
    prog.AddCost(diff_int.ToExpression())
    # RHS = Polynomial((l(z, u_opt) + dJdz.dot(f(z,u_opt)))[0], z)
    # for monomial, coeff in RHS.monomial_to_coefficient_map().items():
    #     # print(f'monomial: {monomial}, coef: {coeff}')
    #     prog.AddConstraint(coeff == 0)

    samples = np.linspace(z_min, z_max)
    for x in samples:
        prog.AddLinearConstraint(
            calc_value_function(x, J_coeff, poly_func) >= 0)

    # J(z0) = 0
    J0 = calc_value_function([params_dict["z0"]], J_coeff, poly_func)
    prog.AddLinearConstraint(J0 == 0)
    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)

    J_star = result.GetSolution(J_coeff_var).reshape(coeff_shape)

    # assert result.is_success()
    print(result.is_success())
    print(result.get_solver_id().name())
    print("Error: ", result.get_optimal_cost())

    plot_value_function(J_star, params_dict, poly_func, poly_deg)

    return J_star


def convex_sampling_hjb_lower_bound(poly_deg, params_dict):
    # Sample for nonnegativity constraint of HJB RHS
    nz = params_dict["nz"]
    coeff_shape = np.ones(nz, dtype=int) * (poly_deg + 1)
    f = params_dict["f"]
    l = params_dict["l"]
    f2 = params_dict["f2"]
    z_max = np.array([1])
    z_min = -z_max
    def poly_func(x, n): return x**n  # monomial(x, n)

    prog = MathematicalProgram()
    J_coeff_var = prog.NewContinuousVariables(np.product(coeff_shape),
                                              "J")
    J_coeff = np.array(J_coeff_var).reshape(coeff_shape)

    z = prog.NewIndeterminates(nz, "z")
    dJdz = calc_dJdz(z, J_coeff, poly_func)
    u_opt = calc_u_opt(dJdz, f2, params_dict["Rinv"])

    RHS = -(l(z, u_opt) + dJdz.dot(f(z, u_opt)))[0]
    samples = np.linspace(z_min, z_max, 500)
    for x in samples:
        prog.AddLinearConstraint(
            calc_value_function(x, J_coeff, poly_func) >= 0)
        constr = RHS.EvaluatePartial(dict(zip(z, x)))
        poly = Polynomial(constr)
        variables, map_var_to_index = sym.ExtractVariablesFromExpression(
            constr)
        Q, b, c = sym.DecomposeQuadraticPolynomial(poly, map_var_to_index)
        try:
            prog.AddQuadraticAsRotatedLorentzConeConstraint(
                Q, b, c, variables, psd_tol=1e-5)
        except:
            pass

    obj = Polynomial(calc_value_function(z, J_coeff, poly_func))
    for i in range(nz):
        obj = obj.Integrate(z[i], z_min[i], z_max[i])
    prog.AddCost(-obj.ToExpression())

    # J(z0) = 0
    J0 = calc_value_function([params_dict["z0"]], J_coeff, poly_func)
    prog.AddLinearConstraint(J0 == 0)
    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)

    J_star = result.GetSolution(J_coeff_var).reshape(coeff_shape)

    # assert result.is_success()
    print(result.is_success())
    print("area: ", result.get_optimal_cost())
    print("# of quadratic constraints: ", len(
        prog.rotated_lorentz_cone_constraints()))

    plot_value_function(J_star, params_dict, poly_func, poly_deg)

    return J_star


def iterative_hjb_regression(poly_deg, params_dict, rho=0):
    # Iterative convex least squares regression
    # Actually no need to add discount factor rho for convergence
    nz = params_dict["nz"]
    coeff_shape = np.ones(nz, dtype=int) * (poly_deg + 1)
    old_coeff = np.zeros(coeff_shape)
    f = params_dict["f"]
    l = params_dict["l"]
    f2 = params_dict["f2"]
    z_max = np.array([1])
    z_min = -z_max
    def poly_func(x, n): return x**n  # monomial(x, n)

    z = np.array([Variable("z")])
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
        diff_int = Polynomial(
            (l(z, u_opt) + dJdz.dot(f(z, u_opt)) - rho * J).squeeze()**2, z)
        for i in range(nz):
            diff_int = diff_int.Integrate(z[i], z_min[i], z_max[i])
        prog.AddQuadraticCost(diff_int.ToExpression(), is_convex=True)

        samples = np.linspace(z_min, z_max)
        for x in samples:
            prog.AddLinearConstraint(
                calc_value_function(x, J_coeff, poly_func) >= 0)

        # J(z0) = 0
        J0 = calc_value_function([params_dict["z0"]], J_coeff, poly_func)
        prog.AddLinearConstraint(J0 == 0)
        # options = SolverOptions()
        # options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        # prog.SetSolverOptions(options)
        solver = GurobiSolver()
        result = solver.Solve(prog)

        J_star = result.GetSolution(J_coeff_var).reshape(coeff_shape)

        assert result.is_success()
        print(result.is_success())
        # print(mp.GetProgramType(prog))
        # print(result.get_solver_id().name())

        if np.allclose(J_star, old_coeff, atol=1e-5):
            print("Error: ", result.get_optimal_cost())
            break
        else:
            # print("Diff: ", np.linalg.norm(J_star-old_coeff))
            old_coeff = J_star

    plot_value_function(J_star, params_dict, poly_func, poly_deg)

    return J_star

def iterative_hjb_sampling_regression(poly_deg, params_dict, rho=0):
    # Iterative convex least squares regression using sampling instead of integration
    nz = params_dict["nz"]
    coeff_shape = np.ones(nz, dtype=int) * (poly_deg + 1)
    old_coeff = np.zeros(coeff_shape)
    f = params_dict["f"]
    l = params_dict["l"]
    f2 = params_dict["f2"]
    z_max = np.array([1])
    z_min = -z_max
    def poly_func(x, n): return x**n  # monomial(x, n)

    z = np.array([Variable("z")])
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

        RHS = (l(z, u_opt) + dJdz.dot(f(z, u_opt)) - rho * J).squeeze()**2
        samples = np.linspace(z_min, z_max)
        for x in samples:
            prog.AddLinearConstraint(
                calc_value_function(x, J_coeff, poly_func) >= 0)
            prog.AddQuadraticCost(RHS.EvaluatePartial(dict(zip(z,[x]))), is_convex=True)
            
        # J(z0) = 0
        J0 = calc_value_function([params_dict["z0"]], J_coeff, poly_func)
        prog.AddLinearConstraint(J0 == 0)
        # options = SolverOptions()
        # options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        # prog.SetSolverOptions(options)
        solver = GurobiSolver()
        result = solver.Solve(prog)

        J_star = result.GetSolution(J_coeff_var).reshape(coeff_shape)

        assert result.is_success()
        print(result.is_success())

        if np.allclose(J_star, old_coeff, atol=1e-5):
            print("Error: ", result.get_optimal_cost())
            break
        else:
            old_coeff = J_star

    plot_value_function(J_star, params_dict, poly_func, poly_deg)

    return J_star


def iterative_hjb_sos_lower_bound(poly_deg, params_dict):
    # Iterative sos hjb lower bound
    nz = params_dict["nz"]
    f = params_dict["f"]
    l = params_dict["l"]
    f2 = params_dict["f2"]
    def poly_func(x, n): return x**n  # monomial(x, n)
    z_max = np.array([1])
    z_min = -z_max

    z = np.array([Variable("z")])
    old_J = Polynomial(np.ones(nz).dot(z))

    for iter in range(100):
        print("Iter: ", iter)
        prog = MathematicalProgram()
        prog.AddIndeterminates(z)
        old_dJdz = old_J.ToExpression().Jacobian(z)
        J = prog.NewFreePolynomial(Variables(z), poly_deg, "J")
        J_expr = J.ToExpression()

        u_opt = calc_u_opt(old_dJdz, f2, params_dict["Rinv"])
        dJdz = J_expr.Jacobian(z)
        RHS = Polynomial((l(z, u_opt) + dJdz.dot(f(z,u_opt)))[0], z)
        prog.AddSosConstraint(RHS.ToExpression())

        # J(z0) = 0
        J0 = J_expr.EvaluatePartial(dict(zip(z, [params_dict["z0"]])))
        prog.AddLinearConstraint(J0 == 0)  
        # Enforce that value function is PD
        prog.AddSosConstraint(J_expr)

        obj = J
        for i in range(nz):
            obj = obj.Integrate(z[i], z_min[i], z_max[i])
        prog.AddCost(-obj.ToExpression())

        # options = SolverOptions()
        # options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        # prog.SetSolverOptions(options)
        result = Solve(prog)
        # print(result.get_solver_id().name())
        # for binding in prog.positive_semidefinite_constraints():
        #     psd_primal = result.GetSolution(binding.variables()).reshape((binding.evaluator().matrix_rows(), binding.evaluator().matrix_rows()))
        #     psd_dual = pydrake.math.ToSymmetricMatrixFromLowerTriangularColumns(result.GetDualSolution(binding))
        #     print(f"primal minimal eigenvalue {np.linalg.eig(psd_primal)[0].min()}")
        #     print(f"dual minimal eigenvalue {np.linalg.eig(psd_dual)[0].min()}")
        # print(prog)
        # assert result.is_success()

        J_star = result.GetSolution(J)

        if J_star.CoefficientsAlmostEqual(old_J, 1e-5):
            break
        # else:
        #     diff = J_star - old_J
        #     coeff_diff = []
        #     for monomial,coeff in diff.monomial_to_coefficient_map().items():
        #         coeff_diff.append(coeff)
        #         # print(f'monomial: {monomial}, coef: {coeff}')
        #     # print("coefficient difference: ", coeff_diff)
        old_J = J_star
        

    dJdz = J_star.ToExpression().Jacobian(z)
    u_star = calc_u_opt(dJdz, f2, params_dict['Rinv'])
    return J_star, u_star, z


def plot_value_function(coeff, params_dict, poly_func, deg):
    x = np.linspace(-1, 1, 100)

    prog = MathematicalProgram()
    z = prog.NewIndeterminates(params_dict["nz"], 'z')
    dJdz = calc_dJdz(z, coeff, poly_func)
    u_opt_expr = calc_u_opt(dJdz, params_dict["f2"], params_dict["Rinv"])
    J = np.zeros(len(x))
    U = np.zeros(len(x))
    for i in range(len(x)):
        J[i] = calc_value_function([x[i]], coeff, poly_func)
    plt.plot(x, J, label="deg. {}".format(deg))
    # plt.savefig("J.png")

def plot_value_function_sos(J_star, u_star, z, poly_deg):
    x = np.linspace(-1, 1, 100)
    J = np.zeros(len(x))
    for i in range(len(x)):
        J[i] = J_star.Evaluate({z[0]: x[i]})
    plt.plot(x, J, label="deg. {}".format(poly_deg))    

def fit_optimal_cost_to_go(poly_deg, J_target, mesh_pts, params_dict):
    n_mesh = len(J_target)
    coeff_shape = np.ones(params_dict["nz"], dtype=int) * (poly_deg + 1)
    def poly_func(x, n): return x**n

    X = []
    for i in range(n_mesh):
        X.append(calc_basis([mesh_pts[i]], coeff_shape, poly_func))
    X = np.array(X)
    J = np.linalg.lstsq(X, J_target)[0].reshape(coeff_shape)
    plot_value_function(J, params_dict, poly_func, poly_deg)


def optimal_cost_to_go():
    x = np.linspace(1e-7, 1, 100)
    J = 2 * (x**2/2 - x**4 + (np.sqrt(x**2 - 4*x**4 +
                                      8*x**6) * (1/32*(-4 + 16*x**2)*np.sqrt(1 - 4*x**2 + 8*x**4) +
                                                 np.arcsinh(1/4 * (-4 + 16*x**2))/(8*np.sqrt(2))))/(
        x * np.sqrt(2*(1 - 4*x**2 + 8*x**4))))
    x = np.concatenate((-np.flip(x), x))
    J = np.concatenate((np.flip(J), J))
    J -= np.min(J)
    return x, J


if __name__ == '__main__':
    params_dict = cubic_setup()
    x_opt, J_opt = optimal_cost_to_go()
    for poly_deg in range(2, 12, 2):
        print("Deg: ", poly_deg)
        convex_sampling_hjb_lower_bound(poly_deg, params_dict)
        # fit_optimal_cost_to_go(poly_deg, J_opt, x_opt, params_dict)
        # J, u, z, = iterative_hjb_sos_lower_bound(poly_deg, params_dict)
        # plot_value_function_sos(J, u, z, poly_deg)

    plt.plot(x_opt, J_opt.T, 'k', label='J*')
    plt.legend()
    plt.savefig("J_convex_sampling_hjb_lower_bound.png")
