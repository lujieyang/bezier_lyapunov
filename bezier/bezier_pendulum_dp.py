import numpy as np
from bezier_operation import *
from pydrake.all import (MathematicalProgram, Variable, ge, CommonSolverOption,
                         Solve, SolverOptions, Variables, Polynomial, le, eq,
                         MosekSolver)
import pydrake.symbolic as sym

def pendulum_dp(deg):
    # [s, c, theta_dot]
    b = 0.1
    x0 = [0, 1, 0]
    f1 = np.zeros([1, 2, 2])
    f2 = np.zeros([2, 1, 2])
    f3 = np.zeros([2, 1, 2])
    g = np.zeros([1, 1, 1])
    l1 = np.zeros([3, 3, 3])
    f1[0, 1, 1] = 1
    f2[1, 0, 1] = -1
    f3[1, 0, 0 ] = 1
    f3[0, 0, 1] = -b
    g[0, 0, 0] = 1
    
    R = np.diag([1]) 

    num_J_degrees = np.array(deg)
    num_var = len(num_J_degrees)
    prog = MathematicalProgram()
    J_var = prog.NewContinuousVariables(np.product(num_J_degrees + 1),
                                        "J")  # Drake is not working for
    # tensor, so vectorize the tensor
    J = np.array(J_var).reshape(num_J_degrees + 1)

    for ind in 2*np.eye(num_var, dtype=int):
        l1[tuple(ind)] = 1

    dJdx = bernstein_derivative(J)

    f_bern = [power_to_bernstein_poly(f) for f in [f1, f2, f3]]
    g_bern = power_to_bernstein_poly(g)
    l1_bern = power_to_bernstein_poly(l1)

    dJdx_f = power_to_bernstein_poly(np.zeros(np.ones(num_var, dtype=int)))
    for dim in range(num_var):
        dJdx_f = bernstein_add(dJdx_f, bernstein_mul(dJdx[dim], f_bern[dim], dtype=Variable))

    dJdx_g = bernstein_mul(dJdx[dim], g_bern, dtype=Variable)
    last_term = -bernstein_mul(dJdx_g, dJdx_g, dtype=Variable)/4
    LHS = bernstein_add(bernstein_add(l1_bern, dJdx_f), last_term)

    J0 = BezierSurface(x0, J)
    prog.AddLinearConstraint(J0 == 0)
    prog.AddLinearConstraint(ge(J_var, 0))

    eq_constraint = ge(LHS, 0)
    for c in eq_constraint.flatten():
        if len(c.GetFreeVariables()) > 0:
            prog.AddConstraint(c)
    
    # it = np.nditer(LHS, flags=['multi_index', 'refs_ok'])
    # for _ in it:
    #     idx = it.multi_index
    #     lhs = LHS[idx]
    #     poly = sym.Polynomial(lhs)
    #     variables, map_var_to_index = sym.ExtractVariablesFromExpression(-lhs)
    #     Q, b, c = sym.DecomposeQuadraticPolynomial(poly, map_var_to_index)
    
    J_int = bernstein_integral(J)
    prog.AddLinearCost(-J_int)

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    print(result.is_success())
    J_opt = np.squeeze(result.GetSolution(J_var)).reshape(num_J_degrees+1)
    return J_opt, -result.get_optimal_cost()


if __name__ == '__main__':
    J, _ = pendulum_dp(4 * np.ones(3, dtype=int))
    x2z = lambda t, td: np.array([np.sin(t), np.cos(t), td])
    plot_energy(J, name="J", x2z=x2z)