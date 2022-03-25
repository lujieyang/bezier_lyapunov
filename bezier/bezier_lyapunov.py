import numpy as np
from bezier_operation import *
from pydrake.all import (MathematicalProgram, Variable, ge, CommonSolverOption,
                         Solve, SolverOptions, Variables, Polynomial, le, eq)
import matplotlib.pyplot as plt


def pendulum_lyapunov(deg, l_deg, alpha=0, eps=1e-3):
    b = 100
    f1 = np.zeros([1, 2, 2])
    f2 = np.zeros([2, 1, 2])
    f3 = np.zeros([2, 1, 2])
    f1[0, 1, 1] = 1
    f2[1, 0, 1] = -1
    f3[1, 0, 0 ] = -1
    f3[0, 0, 1] = -b

    num_V_degrees = np.array(deg)
    num_var = len(num_V_degrees)
    prog = MathematicalProgram()
    Z = tuple(np.zeros(num_var, dtype=int))
    V_var = prog.NewContinuousVariables(np.product(num_V_degrees + 1),
                                        "V")  # Drake is not working for
    # tensor, so vectorize the tensor
    V = np.array(V_var).reshape(num_V_degrees + 1)

    dVdx = bernstein_derivative(V)

    f_bern = [power_to_bernstein_poly(f) for f in [f1, f2, f3]]
    Vdot = power_to_bernstein_poly(np.zeros(np.ones(num_var, dtype=int)))
    for dim in range(num_var):
        Vdot = bernstein_add(Vdot, bernstein_mul(dVdx[dim], f_bern[dim], dtype=Variable))

    x_square = np.zeros(np.ones(num_var, dtype=int) * 3)
    for ind in 2*np.eye(num_var, dtype=int):
        x_square[tuple(ind)] = 1
    # Reshape LHS to vector since prog.AddConstraint doesn't accept tensor
    LHS_V = bernstein_add(V, -eps*power_to_bernstein_poly(x_square)).reshape(-1, 1)

    prog.AddLinearConstraint(V[Z] == 0)
    prog.AddLinearConstraint(ge(LHS_V, 0))

    num_l_degrees = np.array(l_deg)
    l = prog.NewContinuousVariables(np.product(num_l_degrees + 1),
                                    "l") 
    l = np.array(l).reshape(num_l_degrees + 1)
    sc_1 = np.zeros([3, 3, 1])
    sc_1[0, 0, 0] = -1
    sc_1[2, 0, 0] = 1
    sc_1[0, 2, 0] = 1
    s_proc = bernstein_mul(l, power_to_bernstein_poly(sc_1), dtype=Variable)
    # Exponential stability with rate alpha
    LHS = bernstein_add(bernstein_add(Vdot, alpha * V), s_proc)

    neg_constraint = le(LHS, 0)
    # TODO: remove after Drake handles formula True
    for c in neg_constraint.flatten():
        if len(c.GetFreeVariables()) > 0:
            prog.AddLinearConstraint(c)

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    print(result.is_success())
    V_opt = result.GetSolution(V_var).reshape(num_V_degrees+1)
    return V_opt, f_bern

def cubic_lyapunov(deg, alpha=.1, eps=1e-3):
    # \dot x = -x + x^3
    f = np.array([0, -1, 0, 1/4])
    # \dot x = x - 4 * x^3
    # f = np.array([0, 1, 0, -4])

    num_V_degrees = np.array([deg])
    num_var = len(num_V_degrees)
    prog = MathematicalProgram()
    Z = tuple(np.zeros(num_var, dtype=int))
    V = prog.NewContinuousVariables(np.product(num_V_degrees + 1),
                                        "V")  # Drake is not working for
    # tensor, so vectorize the tensor
    V = np.array(V).reshape(num_V_degrees + 1)

    dVdx = bernstein_derivative(V)

    f_bern = [power_to_bernstein_poly(f)]
    Vdot = np.zeros(num_var)
    for dim in range(num_var):
        Vdot = bernstein_add(Vdot, bernstein_mul(dVdx[dim], f_bern[dim], dtype=Variable))

    x_square = np.array([0, 0, 1])
    LHS_V = bernstein_add(V, -eps*power_to_bernstein_poly(x_square))

    prog.AddLinearConstraint(V[Z] == 0)
    prog.AddLinearConstraint(ge(LHS_V, 0))
    # Exponential stability with rate alpha
    LHS = bernstein_add(Vdot, alpha * V)
    neg_constraint = le(LHS, 0)
    # TODO: remove after Drake handles formula True
    for c in neg_constraint.flatten():
        if len(c.GetFreeVariables()) > 0:
            prog.AddLinearConstraint(c)

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    print(result.is_success())
    dVdt = result.GetSolution(Vdot)
    return result.GetSolution(V), f_bern


def cubic_roa_lyapunov(deg, l_deg, alpha=0, eps=1e-3):
    # \dot x = x - 4 * x^3
    f = np.array([0, 1, 0, -4])

    num_V_degrees = np.array([deg])
    num_var = len(num_V_degrees)
    prog = MathematicalProgram()
    Z = tuple(np.zeros(num_var, dtype=int))
    V = prog.NewContinuousVariables(np.product(num_V_degrees + 1),
                                        "V")  # Drake is not working for
    # tensor, so vectorize the tensor
    V = np.array(V).reshape(num_V_degrees + 1)

    dVdx = bernstein_derivative(V)

    f_bern = [power_to_bernstein_poly(f)]
    Vdot = np.zeros(num_var)
    for dim in range(num_var):
        Vdot = bernstein_add(Vdot, bernstein_mul(dVdx[dim], f_bern[dim], dtype=Variable))

    x_square = np.array([0, 0, 1])
    LHS_V = bernstein_add(V, -eps*power_to_bernstein_poly(x_square))

    prog.AddLinearConstraint(V[Z] == 0)
    prog.AddLinearConstraint(ge(LHS_V, 0))
    
    # ROA -0.5<= x <= 0.5
    # roa = np.array([-0.5, 1])
    roa = np.array([-0.25, 0, 1])
    num_l_degrees = np.array(l_deg)
    l = prog.NewContinuousVariables(np.product(num_l_degrees + 1),
                                    "l") 
    l = np.array(l).reshape(num_l_degrees + 1)
    s_proc = bernstein_mul(l, power_to_bernstein_poly(roa), dtype=Variable)
    prog.AddLinearConstraint(ge(l, 0))

    # Exponential stability with rate alpha
    LHS = bernstein_add(-Vdot, s_proc)
    pos_constraint = ge(LHS, 0)
    # TODO: remove after Drake handles formula True
    for c in pos_constraint.flatten():
        if len(c.GetFreeVariables()) > 0:
            prog.AddLinearConstraint(c)

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    print(result.is_success())
    dVdt = result.GetSolution(Vdot)
    return result.GetSolution(V), f_bern


def main_lyapunov():
    degrees = 2
    V, f_bern = cubic_lyapunov(degrees, alpha=0.5)
    # TODO: remove after symbolic::get_constant_value gets exposed to pydrake
    dVdx = bernstein_derivative(V)
    Vdot = bernstein_mul(dVdx[0], f_bern[0])
    V *= 1e3
    Vdot *= 1e3
    plot_bezier(V, -1, 1, label='V')
    plot_bezier(Vdot, -1, 1, label='Vdot')
    plt.legend()
    plt.savefig("lyapnov.png")


if __name__ == '__main__':
    main_lyapunov()
    # V, f_bern = pendulum_lyapunov(4 * np.ones(3, dtype=int), [2, 2, 2],alpha=0)
    # V *=1e3
    # plot_energy(V)
    # cubic_control_affine_dp(10)











