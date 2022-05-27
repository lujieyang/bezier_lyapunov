import time
from utils import *
from cubic_polynomial_fvi import optimal_cost_to_go
from pydrake.all import (Solve, SolverOptions, CommonSolverOption, Polynomial, Variables, ge)

import matplotlib.pyplot as plt

def cubic_batch_setup():
    # f(x) = x - 4x^3 + u
    nz = 1
    nu = 1
    x0 = 0


    def f1(x):
        return x - 4*x**3

    def f2(x):
        return np.ones([x.shape[0], nz, nu])

    Q = np.diag([1])
    R = 1

    return {"nz": nz, "f1": f1, "f2": f2, "Q": Q, "R": R, "x0": x0, "Rinv":np.diag([1/R])}

def convex_sampling_hjb_lower_bound_batch_calc(deg, params_dict, n_mesh=6, sampling_strategy="grid", visualize=True):
    # Sample for nonnegativity constraint of HJB RHS
    nz = params_dict["nz"]
    f1 = params_dict["f1"]
    f2 = params_dict["f2"]
    Q = params_dict["Q"]
    Rinv = params_dict["Rinv"]
    x0 = params_dict["x0"]
    z_max = np.array([1])
    z_min = -z_max

    prog = MathematicalProgram()
    z = prog.NewIndeterminates(nz, "z")
    J = prog.NewFreePolynomial(Variables(z), deg)
    J_decision_variables = np.array(list(J.decision_variables()))
    nJ = len(J_decision_variables)
    calc_basis = construct_monomial_basis_from_polynomial(J, nJ, z)

    J_expr = J.ToExpression()
    dJdz = J_expr.Jacobian(z)

    if sampling_strategy == "random":
        X = np.expand_dims(np.random.uniform(z_min, z_max, n_mesh), axis=1)
    elif sampling_strategy == "grid":
        X = np.linspace(z_min, z_max, n_mesh)
    Z = X

    J_basis = calc_basis(Z)

    dJdz_poly = Polynomial(dJdz[0], z)
    calc_basis_dJdz = construct_monomial_basis_from_polynomial(dJdz_poly, nJ, z)
    dphi_dx = calc_basis_dJdz(Z)
    dPhi_dx = np.expand_dims(dphi_dx, axis=1)

    f2_val = f2(X)
    f2_dPhi_dx = np.matmul(np.transpose(f2_val, (0, 2, 1)), dPhi_dx)
    Q_batch = np.einsum("bki, ii, bik->bik", f2_dPhi_dx, Rinv, f2_dPhi_dx)

    l1 = np.einsum("bi, ij, bj->b", X-x0, Q, X-x0)
    f1_val = f1(X)
    dPhi_dx_f1 = np.matmul(np.transpose(dPhi_dx, (0, 2, 1)), np.expand_dims(f1_val, axis=2)) 

    start_time = time.time()
    for i in range(len(l1)):
        if i % 1000 == 0:
            print("Mesh x0 No.", i)
        c = -l1[i]
        b = -dPhi_dx_f1[i]
        Qin = 2 * Q_batch[i]/4
        try:
            prog.AddQuadraticAsRotatedLorentzConeConstraint(
                Qin, b, c, J_decision_variables, psd_tol=1e-5)
        except:
            pass
    prog.AddLinearConstraint(J_basis, np.zeros(J_basis.shape[0]), np.inf*np.ones(J_basis.shape[0]), J_decision_variables)
    end_time = time.time()
    print("Time for adding constraints: ", end_time-start_time)

    obj = J
    for i in range(nz):
        obj = obj.Integrate(z[i], z_min[i], z_max[i])
    prog.AddCost(-obj.ToExpression())

    # J(z0) = 0
    J0 = J_expr.EvaluatePartial(dict(zip(z, [params_dict["x0"]])))
    prog.AddLinearConstraint(J0 == 0)
    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)

    print("="*10, "Solving","="*20)
    solve_start = time.time()
    result = Solve(prog)
    solve_end = time.time()
    print("Time for solving: ", solve_end-solve_start)

    J_star = result.GetSolution(J)

    # assert result.is_success()
    print(result.is_success())
    print("area: ", result.get_optimal_cost())
    print("# of quadratic constraints: ", len(
        prog.rotated_lorentz_cone_constraints()))

    dJdz = J_star.ToExpression().Jacobian(z)
    if visualize:
        plot_value_function(J_star, z, deg)

    return J_star, z, prog, J_expr

def plot_value_function(J, z, deg):
    x = np.linspace(-1, 1, 100)
    J_val = []
    for i in range(len(x)):
        J_val.append(J.Evaluate({z[0]: x[i]}))
    plt.plot(x, J_val, label="deg. {}".format(deg))

if __name__ == '__main__':
    params_dict = cubic_batch_setup()
    n_mesh = 15
    for deg in range(2, 12, 2):
        convex_sampling_hjb_lower_bound_batch_calc(deg, params_dict, n_mesh, sampling_strategy="random")
    
    x_opt, J_opt = optimal_cost_to_go()
    plt.plot(x_opt, J_opt.T, 'k', label='J*')
    plt.legend()
    plt.savefig("cubic/figures/J_uniform_grid_mesh_{}.png".format(n_mesh))