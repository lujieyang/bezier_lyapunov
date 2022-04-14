from re import I
import numpy as np
from bezier_operation import *
from pydrake.all import (MathematicalProgram, Variable, ge, CommonSolverOption,
                         Solve, SolverOptions, Variables, Polynomial, le, eq)
from pydrake.examples.quadrotor import QuadrotorPlant
import pydrake.symbolic as sym
import matplotlib.pyplot as plt


def quadrotor_control_lyapunov():
    # state = [x, y, s, c, xdot, ydot, thetadot]
    S_IDX = 2
    C_IDX = 3
    XD_IDX = 4
    YD_IDX = 5
    TD_IDX = 6

    p = QuadrotorPlant()
    m = p.m()
    g = p.g()
    x_dim = 7
    u_dim = 2
    zero_dim = np.ones(x_dim, dtype=int)

    s_ind = np.zeros(x_dim, dtype=int)
    s_ind[S_IDX] = 1
    c_ind = np.zeros(x_dim, dtype=int)
    c_ind[C_IDX] = 1
    xdot_ind = np.zeros(x_dim, dtype=int)
    xdot_ind[XD_IDX] = 1
    ydot_ind = np.zeros(x_dim, dtype=int)
    ydot_ind[YD_IDX] = 1
    thetadot_ind = np.zeros(x_dim, dtype=int)
    thetadot_ind[TD_IDX] = 1

    
    f0 = np.zeros(xdot_ind + zero_dim)
    f1 = np.zeros(ydot_ind + zero_dim)
    f2 = np.zeros(c_ind + thetadot_ind + zero_dim)
    f3 = np.zeros(s_ind + thetadot_ind + zero_dim)
    f4 = np.zeros(zero_dim)
    f5 = np.zeros(zero_dim) 
    f6 = np.zeros(zero_dim) 
    f0[tuple(xdot_ind)] = 1
    f1[tuple(ydot_ind)] = 1
    f2[tuple(c_ind + thetadot_ind)] = 1
    f3[tuple(s_ind + thetadot_ind)] = -1
    f5[tuple(zero_dim-1)] = -g

    
    h = np.zeros([x_dim, u_dim], dtype=np.ndarray)
    for i in range(x_dim):
        for j in range(u_dim):
            h[i, j] = np.zeros(zero_dim)
    
    h[4, 0][tuple(s_ind)] = -1/m
    h[4, 1][tuple(s_ind)] = -1/m

    h[5, 0][tuple(c_ind)] = 1/m
    h[5, 1][tuple(c_ind)] = 1/m

    h[6, 0][tuple(zero_dim-1)] = 1/I
    h[6, 1][tuple(zero_dim-1)] = -1/I


def van_der_pol_control_lyapunov(deg, alpha=0, eps=1e-3):
    # f = [x1, 
    #      -x0 - x0^2 * x1 + x1]
    x0 = [0, 0]
    f1 = np.zeros([1, 2])
    f2 = np.zeros([3, 2])
    f1[0, 1] = 1
    f2[1, 0] = -1
    f2[2, 1] = -1
    f2[0, 1] = 1

    num_V_degrees = np.array(deg)
    num_var = len(num_V_degrees)
    prog = MathematicalProgram()
    Z = tuple(np.zeros(num_var, dtype=int))
    V_var = prog.NewContinuousVariables(np.product(num_V_degrees + 1),
                                        "V")  # Drake is not working for
    # tensor, so vectorize the tensor
    V = np.array(V_var).reshape(num_V_degrees + 1)

    dVdx = bernstein_derivative(V)

    f_bern = [power_to_bernstein_poly(f) for f in [f1, f2]]
    Vdot = power_to_bernstein_poly(np.zeros(np.ones(num_var, dtype=int)))
    for dim in range(num_var):
        Vdot = bernstein_add(Vdot, bernstein_mul(dVdx[dim], f_bern[dim], dtype=Variable))

    x_square = np.zeros(np.ones(num_var, dtype=int) * 3)
    for ind in 2*np.eye(num_var, dtype=int):
        x_square[tuple(ind)] = 1
    # Reshape LHS to vector since prog.AddConstraint doesn't accept tensor
    LHS_V = bernstein_add(V, -eps*power_to_bernstein_poly(x_square)).reshape(-1, 1)

    V0 = BezierSurface(x0, V)
    prog.AddLinearConstraint(V0 == 0)
    prog.AddLinearConstraint(ge(LHS_V, 0))

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
    V_opt = result.GetSolution(V_var).reshape(num_V_degrees+1)
    return V_opt, f_bern


def pendulum_control_lyapunov(deg, eps=1e-3):
    # Swing up the pendulum
    # [s, c, theta_dot]
    b = 0.1
    x0 = [0, 1, 0]
    x_dim = 3
    u_dim = 1
    f1 = np.zeros([1, 2, 2])
    f2 = np.zeros([2, 1, 2])
    f3 = np.zeros([2, 1, 2])
    
    f1[0, 1, 1] = 1
    f2[1, 0, 1] = -1
    f3[1, 0, 0 ] = 1  # Upward is the positive axis
    f3[0, 0, 1] = -b
    g = np.zeros([x_dim, u_dim], dtype=np.ndarray)
    for i in range(x_dim):
        for j in range(u_dim):
            g[i, j] = np.zeros([1, 1, 1])
    g[2, 0][0, 0, 0] = 1

    num_V_degrees = np.array(deg)
    num_var = len(num_V_degrees)
    d = np.product(num_V_degrees + 1)
    prog = MathematicalProgram()
    Z = tuple(np.zeros(num_var, dtype=int))
    V_var = prog.NewContinuousVariables(d, "V")  # Drake is not working for
    # tensor, so vectorize the tensor
    V = np.array(V_var).reshape(num_V_degrees + 1)

    u = prog.NewContinuousVariables(u_dim, "u")
    prog.AddConstraint(u.dot(u) <= 1)

    dVdx = bernstein_derivative(V)

    f_bern = [power_to_bernstein_poly(f) for f in [f1, f2, f3]]
    LfV = power_to_bernstein_poly(np.zeros(np.ones(num_var, dtype=int)))
    for dim in range(num_var):
        LfV = bernstein_add(LfV, bernstein_mul(dVdx[dim], f_bern[dim], dtype=Variable))

    # Extract LgM matrix
    LgV = np.zeros(u_dim, dtype=np.ndarray)
    LgM = np.zeros(LfV.size, dtype=np.ndarray)
    LfV_deg = np.array(LfV.shape) - 1
    LfV_flatten = LfV.flatten()
    for j in range(u_dim):
        LgV[j] = np.zeros(np.ones(num_var, dtype=int))
        for i in range(x_dim):
            g_bern = power_to_bernstein_poly(g[i, j])
            LgV[j] = bernstein_add(LgV[j], bernstein_mul(dVdx[i], g_bern, dtype=Variable))
        LgV_deg = np.array(LgV[j].shape) - 1
        LgV_flatten = bernstein_degree_elevation(LgV[j], LfV_deg - LgV_deg).flatten()
        for k in range(len(LgV_flatten)):
            lgv = LgV_flatten[k]
            LgM[k] = np.zeros([u_dim, d])
            variables, map_var_to_index = sym.ExtractVariablesFromExpression(lgv)
            M, v = sym.DecomposeAffineExpression(lgv, map_var_to_index)
            for m in range(len(variables)):
                LgM[k][j, variables[m].get_id()-1] = M[m]
    
    # Extract LfM matrix
    for i in range(len(LfV_flatten)):
        lfv = LfV_flatten[i]
        LfM = np.zeros(d)
        variables, map_var_to_index = sym.ExtractVariablesFromExpression(lfv)
        M, v = sym.DecomposeAffineExpression(lfv, map_var_to_index)
        for m in range(len(variables)):
            LfM[variables[m].get_id()-1] = M[m]
        # prog.AddConstraint(LfM.dot(V_var) <= np.linalg.norm(LgM[i]@V_var))
        prog.AddConstraint(LfM.dot(V_var) + V_var.dot(LgM[i].T@u)<=0)
    

    x_square = np.zeros([3, 3, 3])
    x_square[0, 0, 0] = 1
    x_square[2, 0, 0] = 1
    x_square[0, 2, 0] = 1
    x_square[0, 1, 0] = -2
    x_square[0, 0, 2] = 1
    # Reshape LHS to vector since prog.AddConstraint doesn't accept tensor
    LHS_V = bernstein_add(V, -eps*power_to_bernstein_poly(x_square)).reshape(-1, 1)

    V0 = BezierSurface(x0, V)
    prog.AddLinearConstraint(V0 == 0)
    prog.AddLinearConstraint(ge(LHS_V, 0))

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    print(result.is_success())
    print(result.get_solver_id().name())
    V_opt = result.GetSolution(V_var).reshape(num_V_degrees+1)
    return V_opt

    

if __name__ == '__main__':
    pendulum_control_lyapunov(2 * np.ones(3, dtype=int))