import sys
sys.path.append("../underactuated")

import os

import numpy as np
from scipy.integrate import quad
import mcint
from utils import extract_polynomial_coeff_dict, calc_u_opt , save_polynomial, load_polynomial
import pickle
from pydrake.all import (MathematicalProgram, Variables, Expression, Solve, Polynomial, SolverOptions, CommonSolverOption, 
Linearize, LinearQuadraticRegulator, MakeVectorVariable, MosekSolver)
from underactuated.quadrotor2d import Quadrotor2D

import matplotlib.pyplot as plt
from matplotlib import cm

import matplotlib
matplotlib.use('Agg')

# Given the degree for the approximate value function and the polynomials
# in the S procedure, solves the SOS and returns the approximate value function
# (together with the objective of the SOS program).
def quadrotor2d_sos_lower_bound(deg, objective="integrate_ring", visualize=False, test=False):
    nz = 7
    nx = 6
    nu = 2
    
    quadrotor = Quadrotor2D()
    m = quadrotor.mass
    g = quadrotor.gravity
    r = quadrotor.length
    I = quadrotor.inertia
    u0 = m * g / 2. * np.array([1, 1])
    # Map from original state to augmented state.
    # Uses sympy to be able to do symbolic integration later on.
    # x = (x, y, theta, xdot, ydot, thetadot)
    # z = (x, y, s, c, xdot, ydot, thetadot)
    x2z = lambda x : np.array([x[0], x[1], np.sin(x[2]), np.cos(x[2]), x[3], x[4], x[5]])

    def f(z, u, dtype=Expression):
        assert len(z) == nz
        assert len(u) == nu
        s = z[2]
        c = z[3]
        thetadot = z[-1]
        f_val = np.zeros(nz, dtype=dtype)
        f_val[:2] = z[4:6]
        f_val[2] = thetadot * c
        f_val[3] = -thetadot * s
        f_val[4] = -s/m*(u[0]+u[1])
        f_val[5] = c/m*(u[0]+u[1])-g
        f_val[6] = r/I*(u[0]-u[1])
        return f_val

    def f2(z, dtype=Expression):
        assert len(z) == nz
        s = z[2]
        c = z[3]
        f2_val = np.zeros([nz, nu], dtype=dtype)
        f2_val[4,:] = -s/m*np.ones(nu)
        f2_val[5,:] = c/m*np.ones(nu)
        f2_val[6,:] = r/I*np.array([1, -1])
        return f2_val
    
    # State limits (region of state space where we approximate the value function).
    z_max = np.array([1, 1, np.sin(np.pi/2), 1, 1, 1, 1])
    z_min = np.array([-1, -1, -np.sin(np.pi/2), 0, -1, -1, -1])

    # Equilibrium point in both the system coordinates.
    x0 = np.zeros(nx)
    z0 = x2z(x0)
    z0[np.abs(z0)<=1e-6] = 0
        
    # Quadratic running cost in augmented state.
    Q = np.diag([10, 10, 10, 10, 1, 1, r/(2*np.pi)])
    R = np.array([[0.1, 0.05], [0.05, 0.1]])
    def l_cost(z, u):
        return (z - z0).dot(Q).dot(z - z0) + (u-u0).dot(R).dot(u-u0)

    Rinv = np.linalg.inv(R)

    if test:
        return nz, f, f2, x2z, Rinv, z0, u0

    xytheta_idx = [0, 1, 4, 5, 6]

    # Set up optimization.
    prog = MathematicalProgram()
    z = prog.NewIndeterminates(nz, 'z')
    u = prog.NewIndeterminates(nu, 'u')
    J = prog.NewFreePolynomial(Variables(z), deg)
    J_expr = J.ToExpression()

    # Maximize volume beneath the value function.
    if objective=="integrate_all":
        obj = J
        for i in range(nz):
            obj = obj.Integrate(z[i], z_min[i], z_max[i])
        prog.AddCost(-obj.ToExpression())
    elif objective=="integrate_ring":
        obj = J
        for i in xytheta_idx:
            obj = obj.Integrate(z[i], z_min[i], z_max[i])
        c_r = 1
        cost = 0
        for monomial,coeff in obj.monomial_to_coefficient_map().items(): 
            s1_deg = monomial.degree(z[2]) 
            c1_deg = monomial.degree(z[3])
            monomial_int1 = quad(lambda x: np.sin(x)**s1_deg * np.cos(x)**c1_deg, -np.pi, np.pi)[0]
            if np.abs(monomial_int1) <=1e-5:
                monomial_int1 = 0
            cost += monomial_int1 * coeff
        poly = Polynomial(cost)
        cost_coeff = [c.Evaluate() for c in poly.monomial_to_coefficient_map().values()]
        # Make the numerics better
        prog.AddLinearCost(-c_r * cost/np.max(np.abs(cost_coeff)))

    # Enforce Bellman inequality.
    f_val = f(z, u)
    J_dot = J_expr.Jacobian(z).dot(f_val)
    LHS = J_dot + l_cost(z, u)

    # lam_deg = Polynomial(LHS).TotalDegree() - 2
    # S procedure for s^2 + c^2 = 1.
    lam = prog.NewFreePolynomial(Variables(u), 2).ToExpression()
    S_ring = lam * (z[2]**2 + z[3]**2 - 1)
    if deg >= 0:
        S_Jdot = 0
        for i in np.arange(nz):
            lam = prog.NewSosPolynomial(Variables(u), 2)[0].ToExpression()
            S_Jdot += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
        prog.AddSosConstraint(LHS + S_ring + S_Jdot)
    else:
        prog.AddSosConstraint(LHS + S_ring)

    # Enforce that value function is PD
    if deg >= 0:
        lam_r = prog.NewFreePolynomial(Variables(z), deg-2).ToExpression()
        S_r = lam_r * (z[2]**2 + z[3]**2 - 1)
        S_J = 0
        for i in np.arange(nz):
            lam = prog.NewSosPolynomial(Variables(z), deg-2)[0].ToExpression()
            S_J += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
        prog.AddSosConstraint(J_expr + S_J + S_r)
    else:
        prog.AddSosConstraint(J_expr)

    # J(z0) = 0.
    J0 = J_expr.EvaluatePartial(dict(zip(z, z0)))
    prog.AddLinearConstraint(J0 == 0)

    # Solve and retrieve result.
    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    assert result.is_success()
    J_star = Polynomial(result.GetSolution(J_expr)).RemoveTermsWithSmallCoefficients(1e-6)

    os.makedirs("quadrotor2d/data/{}".format(z_max), exist_ok=True)
    os.makedirs("quadrotor2d/figures/{}".format(z_max), exist_ok=True)

    if visualize:
        plot_value_function(J_star, z, z_max, u0, file_name="paper/lower_bound_{}_{}".format(objective, deg), plot_states="xy", u_index=0)
    return J_star, z, z_max

def plot_value_function(J_star, z, z_max, u0, file_name="", plot_states="xy", u_index=0, actuator_saturate=False):
    nz = 7
    x_max = np.zeros(6)
    x_max[:2] = z_max[:2]
    x_max[2] = np.pi/2
    x_max[3:] = z_max[4:]
    x_min = -x_max

    dJdz = J_star.ToExpression().Jacobian(z)

    nz, f, f2, x2z, Rinv, z0, u0 = quadrotor2d_sos_lower_bound(2, test=True)

    zero_vector = np.zeros(51*51)
    if plot_states == "xtheta":
        X1, THETA = np.meshgrid(np.linspace(x_min[0], x_max[0], 51),
                        np.linspace(x_min[2], x_max[2], 51))
        X = np.vstack((X1.flatten(), zero_vector, THETA.flatten(), zero_vector, zero_vector, zero_vector))
        ylabel="theta"
    elif plot_states == "xy":
        X1, Y = np.meshgrid(np.linspace(x_min[0], x_max[0], 51),
                        np.linspace(x_min[1], x_max[1], 51))
        X = np.vstack((X1.flatten(), Y.flatten(), zero_vector, zero_vector, zero_vector, zero_vector))
        ylabel="y"

    Z = x2z(X)
    J = np.zeros(Z.shape[1])
    U = np.zeros(Z.shape[1])
    for i in range(Z.shape[1]):
        z_val = Z[:, i]
        J[i] = J_star.Evaluate(dict(zip(z, z_val)))
        f2_val = f2(z_val, dtype=float)
        dJdz_val = np.zeros(nz)
        for n in range(nz): 
            dJdz_val[n] = dJdz[n].Evaluate(dict(zip(z, z_val)))
        U[i] = calc_u_opt(dJdz_val, f2_val, Rinv)[u_index] + u0[u_index]
        if actuator_saturate:
            U[i] = np.clip(U[i], 0, 2.5*u0[0])

    fig = plt.figure(figsize=(9, 4))
    ax = fig.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.set_title("Cost-to-Go")
    im = ax.imshow(J.reshape(X1.shape),
            cmap=cm.jet, aspect='auto',
            extent=(x_min[0], x_max[0], x_max[2], x_min[2]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.savefig("quadrotor2d/figures/{}_{}.png".format(file_name, plot_states))

    fig = plt.figure(figsize=(9, 4))
    ax = fig.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.set_title("Policy")
    im = ax.imshow(U.reshape(X1.shape),
            cmap=cm.jet, aspect='auto',
            extent=(x_min[0], x_max[0], x_max[2], x_min[2]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.savefig("quadrotor2d/figures/{}_policy_{}_u{}.png".format(file_name, plot_states, u_index+1))

def quadrotor2d_constrained_lqr(nz=7, nu=2):
    quadrotor = Quadrotor2D()
    m = quadrotor.mass
    g = quadrotor.gravity
    r = quadrotor.length
    I = quadrotor.inertia
    A = np.zeros([nz, nz])
    B = np.zeros([nz, nu])
    A[0, 4] = 1
    A[1, 5] = 1
    A[2, 6] = 1
    A[4, 2] = -g
    A[5, 3] = g
    B[5, :] = np.ones(nu)/m
    B[6, :] = r/I*np.array([1, -1])
    F = np.zeros(nz)
    F[3] = 1
    Q = np.diag([10, 10, 10, 10, 1, 1, r/(2*np.pi)])
    R = np.array([[0.1, 0.05], [0.05, 0.1]])
    K, S = LinearQuadraticRegulator(A, B, Q, R, F=F.reshape(1, nz))
    return K, S

def find_regional_lyapunov(z, z0, f_cl, V_deg, ball_size=0.1, eps=1e-3):
    prog = MathematicalProgram()
    prog.AddIndeterminates(z)
    V = prog.NewFreePolynomial(Variables(z), V_deg).ToExpression()

    lam_r_deg = V_deg - 2
    lam_r = prog.NewFreePolynomial(Variables(z), lam_r_deg).ToExpression()
    S_r = lam_r*(z[2]**2+z[3]**2-1)

    lam_deg = V_deg-2
    lam = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_deg/2)*2))[0].ToExpression()
    S_ball = lam*((z-z0).dot(z-z0)-ball_size**2)
    prog.AddSosConstraint(V -eps*(z-z0).dot(z-z0) + S_ball + S_r)

    V0 = V.EvaluatePartial(dict(zip(z, z0)))
    prog.AddLinearConstraint(V0 == 0)

    V_dot = V.Jacobian(z).dot(f_cl)
    lam_r = prog.NewFreePolynomial(Variables(z), lam_r_deg).ToExpression()
    S_r = lam_r*(z[2]**2+z[3]**2-1)
    lam = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_deg/2)*2))[0].ToExpression()
    S_ball = lam*((z-z0).dot(z-z0)-ball_size**2)
    prog.AddSosConstraint(-V_dot -eps*V + S_ball + S_r)  

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    assert result.is_success()

    V_candidate = result.GetSolution(V)
    V_candidate = Polynomial(V_candidate).RemoveTermsWithSmallCoefficients(1e-6).ToExpression()
    return V_candidate

def quadrotor2d_lqr_ROA(controller="linear", find_regional=False):
    nz, f, f2, x2z, Rinv, z0, u0 = quadrotor2d_sos_lower_bound(2, test=True)
    prog = MathematicalProgram()
    z = prog.NewIndeterminates(nz, "z")
    K, S = quadrotor2d_constrained_lqr()
    if controller == 'linear':
        u_star = -K @ (z-z0) + u0
    else:
        f2_val = f2(z)
        u_star = - .5 * Rinv.dot(f2_val.T).dot(dVdz.T)
    f_val = f(z, u_star)
    if find_regional:
        V_deg = 4
        V = find_regional_lyapunov(z, z0, f_val, V_deg, ball_size=1)
    else:
        V_deg = 2
        V = (z-z0).dot(S).dot(z-z0) + 1e-4 * (z-z0).dot(z-z0)
    dVdz = V.Jacobian(z)
    V_dot = dVdz.dot(f_val)
    lhs_deg = V_deg + 2
    lam_deg = lhs_deg - Polynomial(V_dot).TotalDegree()
    lam_r_deg = lhs_deg - 2
    lam_r = prog.NewFreePolynomial(Variables(z), lam_r_deg).ToExpression()
    lam = prog.NewFreePolynomial(Variables(z), lam_deg).ToExpression()
    rho = prog.NewContinuousVariables(1, 'rho')[0]

    prog.AddSosConstraint((z-z0).dot(z-z0)*(V - rho) - lam*V_dot + lam_r*(z[2]**2+z[3]**2-1))

    prog.AddLinearCost(-rho)

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    assert result.is_success()

    print("rho: ", result.GetSolution(rho))
    return result.GetSolution(rho)

def quadrotor2d_upper_bound_ROA():
    nz, f, f2, x2z, Rinv, z0, u0 = quadrotor2d_sos_lower_bound(2, test=True)
    prog = MathematicalProgram()
    z = prog.NewIndeterminates(nz, "z")
    V = load_polynomial(z, "quadrotor2d/data/J_upper_bound_deg_2.pkl")
    dVdz = V.Jacobian(z)
    f2_val = f2(z)
    u_star = - .5 * Rinv.dot(f2_val.T).dot(dVdz.T) + u0
    f_val = f(z, u_star)
    V_dot = dVdz.dot(f_val)

    lhs_deg = 2 + 2
    lam_deg = lhs_deg - Polynomial(V_dot).TotalDegree()
    lam_r_deg = lhs_deg - 2
    lam_r = prog.NewFreePolynomial(Variables(z), lam_r_deg).ToExpression()
    lam = prog.NewFreePolynomial(Variables(z), lam_deg).ToExpression()
    rho = prog.NewContinuousVariables(1, 'rho')[0]

    prog.AddSosConstraint((z-z0).dot(z-z0)*(V - rho) - lam*V_dot + lam_r*(z[2]**2+z[3]**2-1))

    prog.AddLinearCost(-rho)

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    assert result.is_success()

    rho_star = result.GetSolution(rho)
    print("rho: ", rho_star)
    return rho_star

def quadrotor2d_lqr_ROA_line_search(controller="linear"):
    nz, f, f2, x2z, Rinv, z0, u0 = quadrotor2d_sos_lower_bound(2, test=True)

    z = MakeVectorVariable(nz, "z")
    K, S = quadrotor2d_constrained_lqr()
    V = (z-z0).dot(S).dot(z-z0) + 1e-4 * (z-z0).dot(z-z0)
    dVdz = V.Jacobian(z)
    if controller == 'linear':
        u_star = -K @ (z-z0) + u0
    else:
        f2_val = f2(z)
        u_star = - .5 * Rinv.dot(f2_val.T).dot(dVdz.T)
    f_val = f(z, u_star)
    V_dot = dVdz.dot(f_val)

    r_deg = 4
    lhs_deg = Polynomial(V_dot).TotalDegree()
    lam_deg = lhs_deg - 2 + r_deg
    lam_deg = int(np.ceil(lam_deg/2)*2)
    lam_r_deg = lhs_deg - 2 + r_deg

    
    def verify(rho):
        prog = MathematicalProgram()
        prog.AddIndeterminates(z)
        r = prog.NewSosPolynomial(Variables(z), r_deg)[0].ToExpression()
        lam_r = prog.NewFreePolynomial(Variables(z), lam_r_deg).ToExpression()
        S_r = lam_r*(z[2]**2+z[3]**2-1)
        lam = prog.NewSosPolynomial(Variables(z), lam_deg)[0].ToExpression()
        
        prog.AddSosConstraint(lam*(V - rho) - (1+r)*V_dot + S_r)

        options = SolverOptions()
        options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        # options.SetOption(MosekSolver.id(), "writedata", "quadrotor2d_roa_line_search.task.gz")
        prog.SetSolverOptions(options)
        result = Solve(prog)

        return result.is_success()

    rho = 0.00039
    rho_step = 1e-7
    while True:
        if verify(rho):
            rho += rho_step
        else:
            rho -= rho_step
            break

    print("rho: ", rho)
    return rho

def quadrotor2d_sos_upper_bound(deg, objective="integrate_ring", visualize=False, actuator_saturate=True):
    nz = 7
    nx = 6
    nu = 2
    
    quadrotor = Quadrotor2D()
    m = quadrotor.mass
    g = quadrotor.gravity
    r = quadrotor.length
    I = quadrotor.inertia
    u0 = m * g / 2. * np.array([1, 1])
    u_max = 2.5 * u0
    u_min = np.zeros(nu)
    # Map from original state to augmented state.
    # Uses sympy to be able to do symbolic integration later on.
    # x = (x, y, theta, xdot, ydot, thetadot)
    # z = (x, y, s, c, xdot, ydot, thetadot)
    x2z = lambda x : np.array([x[0], x[1], np.sin(x[2]), np.cos(x[2]), x[3], x[4], x[5]])

    def f(z, u, dtype=Expression):
        assert len(z) == nz
        assert len(u) == nu
        s = z[2]
        c = z[3]
        thetadot = z[-1]
        f_val = np.zeros(nz, dtype=dtype)
        f_val[:2] = z[4:6]
        f_val[2] = thetadot * c
        f_val[3] = -thetadot * s
        f_val[4] = -s/m*(u[0]+u[1])
        f_val[5] = c/m*(u[0]+u[1])-g
        f_val[6] = r/I*(u[0]-u[1])
        return f_val

    def f2(z, dtype=Expression):
        assert len(z) == nz
        s = z[2]
        c = z[3]
        f2_val = np.zeros([nz, nu], dtype=dtype)
        f2_val[4,:] = -s/m*np.ones(nu)
        f2_val[5,:] = c/m*np.ones(nu)
        f2_val[6,:] = r/I*np.array([1, -1])
        return f2_val
    
    # State limits (region of state space where we approximate the value function).
    z_max = np.array([1, 1, np.sin(np.pi/2), 1, 1, 1, 1])
    z_min = np.array([-1, -1, -np.sin(np.pi/2), 0, -1, -1, -1])
    assert (z_min <= z_max).all()

    # Equilibrium point in both the system coordinates.
    x0 = np.zeros(nx)
    z0 = x2z(x0)
    z0[np.abs(z0)<=1e-6] = 0
        
    # Quadratic running cost in augmented state.
    Q = np.diag([10, 10, 10, 10, 1, 1, r/(2*np.pi)])
    R = np.array([[0.1, 0.05], [0.05, 0.1]])
    def l_cost(z, u):
        return (z - z0).dot(Q).dot(z - z0) + (u- u0).dot(R).dot(u - u0)

    Rinv = np.linalg.inv(R)

    xytheta_idx = [0, 1, 4, 5, 6]

    # Set up optimization.        
    prog = MathematicalProgram()
    z = prog.NewIndeterminates(nz, 'z')
    K, _ = quadrotor2d_constrained_lqr()
    u_fixed = -K @ (z-z0) + u0
    J = prog.NewFreePolynomial(Variables(z), deg)
    J_expr = J.ToExpression()

    a = prog.NewSosPolynomial(Variables(z), deg)[0]

    # Minimize volume beneath the a(x).
    obj = a
    for i in xytheta_idx:
        obj = obj.Integrate(z[i], z_min[i], z_max[i])
    c_r = 1
    cost = 0
    for monomial,coeff in obj.monomial_to_coefficient_map().items(): 
        s1_deg = monomial.degree(z[2]) 
        c1_deg = monomial.degree(z[3])
        monomial_int1 = quad(lambda x: np.sin(x)**s1_deg * np.cos(x)**c1_deg, -np.pi/2, np.pi/2)[0]
        if np.abs(monomial_int1) <=1e-5:
            monomial_int1 = 0
        cost += monomial_int1 * coeff
    poly = Polynomial(cost)
    cost_coeff = [c.Evaluate() for c in poly.monomial_to_coefficient_map().values()]
    # Make the numerics better
    a_cost = prog.AddLinearCost(c_r * cost/np.max(np.abs(cost_coeff)))

    # Enforce Bellman inequality.
    f_val = f(z, u_fixed)
    J_dot = J_expr.Jacobian(z).dot(f_val)
    LHS = a.ToExpression() - J_dot - l_cost(z, u_fixed)

    lam_deg = Polynomial(LHS).TotalDegree() - 2
    # S procedure for s^2 + c^2 = 1.
    lam = prog.NewFreePolynomial(Variables(z), lam_deg).ToExpression()
    S_ring = lam * (z[2]**2 + z[3]**2 - 1)
    if deg >= 0:
        S_Jdot = 0
        # Also constrain theta to be in [-pi/2, pi/2]
        for i in np.arange(nz):
            lam = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_deg/2)*2))[0].ToExpression()
            S_Jdot += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
        if not actuator_saturate:
            prog.AddSosConstraint(LHS + S_ring + S_Jdot)
    else:
        prog.AddSosConstraint(LHS + S_ring)

    # Enforce that value function is PD
    if deg >= 0:
        lam_r = prog.NewFreePolynomial(Variables(z), deg-2).ToExpression()
        S_r = lam_r * (z[2]**2 + z[3]**2 - 1)
        S_J = 0
        # Also constrain theta to be in [-pi/2, pi/2]
        for i in np.arange(nz):
            lam = prog.NewSosPolynomial(Variables(z), deg-2)[0].ToExpression()
            S_J += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
        prog.AddSosConstraint(J_expr + S_J + S_r)
    else:
        prog.AddSosConstraint(J_expr)

    # Enforce l(x,u)-a(x) is PD
    u = prog.NewIndeterminates(nu, 'u')
    if deg >= 0:
        lam_r_la = prog.NewFreePolynomial(Variables(z), deg-2).ToExpression()
        S_r_la = lam_r_la * (z[2]**2 + z[3]**2 - 1)
        S_la = 0
        # Also constrain theta to be in [-pi/2, pi/2]
        for i in np.arange(nz):
            lam = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
            S_la += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
        prog.AddSosConstraint(l_cost(z,u) - a.ToExpression() + S_la + S_r_la)
    else:
        prog.AddSosConstraint(l_cost(z,u) - a.ToExpression())

    # J(z0) = 0.
    J0 = J_expr.EvaluatePartial(dict(zip(z, z0)))
    prog.AddLinearConstraint(J0 == 0)

    # Actuator saturation
    if actuator_saturate:
        LHS_limits = []
        Su_limits = []
        S_ring_limits = []
        S_Jdot_limits = []   
        for k in range(3): #(-inf, u_min),[u_min, u_max], (u_max, inf) 
            u_limit = np.zeros(nu, dtype=Expression)
            Su_limit = 0
            if k == 0:
                u_limit[0] = u_min[0]
            elif k == 1:
                u_limit[0] = u_fixed[0]
            elif k == 2:
                u_limit[0] = u_max[0]
            for n in range(3):
                if n == 0:
                    u_limit[1] = u_min[1]
                elif n == 1:
                    u_limit[1] = u_fixed[1]
                elif n == 2:
                    u_limit[1] = u_max[1]

                f_limit = f(z, u_limit)
                J_dot_limit = J_expr.Jacobian(z).dot(f_limit)
                LHS_limit = a.ToExpression() - J_dot_limit - l_cost(z, u_limit)
                LHS_limits.append(LHS_limit)

                lam_u_deg = Polynomial(LHS_limit).TotalDegree() - np.max([Polynomial(u_limit[0]).TotalDegree(), Polynomial(u_limit[1]).TotalDegree()])
                if k == 0:
                    lam_u = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                    Su_limit += lam_u*(u_fixed[0] - u_min[0])
                elif k == 1:
                    lam_u_min = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                    lam_u_max = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                    Su_limit += lam_u_max*(u_fixed[0] - u_max[0]) + lam_u_min*(u_min[0] - u_fixed[0])
                elif k == 2:
                    lam_u = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                    Su_limit += lam_u*(u_max[0] - u_fixed[0])

                if n == 0:
                    lam_u = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                    Su_limit += lam_u*(u_fixed[1] - u_min[1])
                elif n == 1:
                    lam_u_min = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                    lam_u_max = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                    Su_limit += lam_u_max*(u_fixed[1] - u_max[1]) + lam_u_min*(u_min[1] - u_fixed[1])
                elif n == 2:
                    lam_u = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                    Su_limit += lam_u*(u_max[1] - u_fixed[1])

                lam_limit_deg = Polynomial(LHS_limit).TotalDegree() - 2
                lam_23 = prog.NewFreePolynomial(Variables(z), lam_limit_deg).ToExpression()
                S_ring_limit = lam_23 * (z[2]**2 + z[3]**2 - 1)
                S_Jdot_limit = 0
                for i in np.arange(nz):
                    lam = prog.NewSosPolynomial(Variables(z), lam_limit_deg)[0].ToExpression()
                    S_Jdot_limit += lam*(z[i]-z_max[i])*(z[i]-z_min[i])

                Su_limits.append(Su_limit)
                S_ring_limits.append(S_ring_limit)
                S_Jdot_limits.append(S_Jdot_limit)
                prog.AddSosConstraint(LHS_limit + S_ring_limit + S_Jdot_limit + Su_limit)

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    assert result.is_success()
    a_star = result.GetSolution(a).RemoveTermsWithSmallCoefficients(1e-6).ToExpression()
    print("a(x): ", a_star)
    LHS_a_star = result.GetSolution(LHS)
    if actuator_saturate:
        LHS_limits_a_star = [result.GetSolution(x) for x in LHS_limits]
    
    prog.RemoveCost(a_cost)

    # Maximize volume beneath the value function.
    if objective=="integrate_all":
        obj = J
        for i in range(nz):
            obj = obj.Integrate(z[i], z_min[i], z_max[i])
        prog.AddCost(-obj.ToExpression())
    elif objective=="integrate_ring":
        obj = J
        for i in xytheta_idx:
            obj = obj.Integrate(z[i], z_min[i], z_max[i])
        c_r = 1
        cost = 0
        for monomial,coeff in obj.monomial_to_coefficient_map().items(): 
            s1_deg = monomial.degree(z[2]) 
            c1_deg = monomial.degree(z[3])
            monomial_int1 = quad(lambda x: np.sin(x)**s1_deg * np.cos(x)**c1_deg, -np.pi/2, np.pi/2)[0]
            if np.abs(monomial_int1) <=1e-5:
                monomial_int1 = 0
            cost += monomial_int1 * coeff
        poly = Polynomial(cost)
        cost_coeff = [c.Evaluate() for c in poly.monomial_to_coefficient_map().values()]
        # Make the numerics better
        prog.AddLinearCost(c_r * cost/np.max(np.abs(cost_coeff)))

    # Enforce Bellman inequality.
    if actuator_saturate:
        for i in range(len(Su_limits)):
                prog.AddSosConstraint(LHS_limits_a_star[i] + S_ring_limits[i] + S_Jdot_limits[i] + Su_limits[i])
    else:
        prog.AddSosConstraint(LHS_a_star + S_ring + S_Jdot)

    # J(z0) = 0.
    J0 = J_expr.EvaluatePartial(dict(zip(z, z0)))
    prog.AddLinearConstraint(J0 == 0)

    # Solve and retrieve result.
    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    assert result.is_success()
    J_star = Polynomial(result.GetSolution(J_expr)).RemoveTermsWithSmallCoefficients(1e-6)
    l_val = Polynomial(result.GetSolution(l_cost(z, u_fixed)))

    os.makedirs("quadrotor2d/data/{}".format(z_max), exist_ok=True)
    os.makedirs("quadrotor2d/figures/{}".format(z_max), exist_ok=True)

    if visualize:
        plot_value_function(J_star, z, z_max, u0, file_name="{}/upper_bound_constrained_lqr_{}_{}".format(z_max, objective, deg), plot_states="xy", u_index=0, actuator_saturate=actuator_saturate)
    return J_star, z, z_max

def quadrotor2d_sos_iterative_upper_bound_ROA(deg, objective="integrate_ring", visualize=False, actuator_saturate=False):
    nz = 7
    nx = 6
    nu = 2
    
    quadrotor = Quadrotor2D()
    m = quadrotor.mass
    g = quadrotor.gravity
    r = quadrotor.length
    I = quadrotor.inertia
    u0 = m * g / 2. * np.array([1, 1])
    u_max = 2.5 * u0
    u_min = np.zeros(nu)
    # Map from original state to augmented state.
    # x = (x, y, theta, xdot, ydot, thetadot)
    # z = (x, y, s, c, xdot, ydot, thetadot)
    x2z = lambda x : np.array([x[0], x[1], np.sin(x[2]), np.cos(x[2]), x[3], x[4], x[5]])

    def f(z, u, dtype=Expression):
        assert len(z) == nz
        assert len(u) == nu
        s = z[2]
        c = z[3]
        thetadot = z[-1]
        f_val = np.zeros(nz, dtype=dtype)
        f_val[:2] = z[4:6]
        f_val[2] = thetadot * c
        f_val[3] = -thetadot * s
        f_val[4] = -s/m*(u[0]+u[1])
        f_val[5] = c/m*(u[0]+u[1])-g
        f_val[6] = r/I*(u[0]-u[1])
        return f_val

    def f2(z, dtype=Expression):
        assert len(z) == nz
        s = z[2]
        c = z[3]
        f2_val = np.zeros([nz, nu], dtype=dtype)
        f2_val[4,:] = -s/m*np.ones(nu)
        f2_val[5,:] = c/m*np.ones(nu)
        f2_val[6,:] = r/I*np.array([1, -1])
        return f2_val
    
    # State limits (region of state space where we approximate the value function).
    z_max = np.array([0.03, 0.03, np.sin(np.pi/2), 0.03, 0.03, 0.03, 0.03])
    z_min = np.array([-0.03, -0.03, -np.sin(np.pi/2), 0, -0.03, -0.03, -0.03])

    os.makedirs("quadrotor2d/data/saturation/{}".format(z_max), exist_ok=True)
    os.makedirs("quadrotor2d/figures/saturation/{}".format(z_max), exist_ok=True)

    # Equilibrium point in both the system coordinates.
    x0 = np.zeros(nx)
    z0 = x2z(x0)
    z0[np.abs(z0)<=1e-6] = 0
        
    # Quadratic running cost in augmented state.
    Q = np.diag([10, 10, 10, 10, 1, 1, r/(2*np.pi)])
    R = np.array([[0.1, 0.05], [0.05, 0.1]])
    def l_cost(z, u):
        return (z - z0).dot(Q).dot(z - z0) + (u- u0).dot(R).dot(u - u0)

    Rinv = np.linalg.inv(R)

    xytheta_idx = [0, 1, 4, 5, 6]
       
    def search_a(J_star, u_fixed, rho, integrate_region="sublevel_set"): 
        print("="*10, "Searching for a", "="*20)
        prog = MathematicalProgram()
        prog.AddIndeterminates(z)
        a = prog.NewSosPolynomial(Variables(z), deg)[0]

        J_expr = J_star

        def sampler():
            while True:
                x = np.random.uniform(z_min[0], z_max[0])
                y = np.random.uniform(z_min[1], z_max[1])
                theta = np.random.uniform(0, np.pi/25)
                xd = np.random.uniform(z_min[-3], z_max[-3])
                yd = np.random.uniform(z_min[-2], z_max[-2])
                thetad = np.random.uniform(z_min[-1], z_max[-1])
                z_val = x2z([x, y, theta, xd, yd, thetad])
                if J_star.Evaluate(dict(zip(z, z_val))) <= rho:
                    yield (x, y, theta, xd, yd, thetad)
        
        def integrate_subelevel_set_monte_carlo(monomial_deg, n_samples=10000):
            assert len(monomial_deg) == nz
            def integrand(x):
                assert len(x) == nx
                return x[0]**monomial_deg[0] * x[1]**monomial_deg[1] * \
                np.sin(x[2])**monomial_deg[2] * np.cos(x[2])**monomial_deg[3] * \
                    x[3]**monomial_deg[4] * x[4]**monomial_deg[5]* x[5]**monomial_deg[6]

            result, error = mcint.integrate(integrand, sampler(), measure=1, n=n_samples)
            return result 

        # Minimize volume beneath the a(x).
        if integrate_region == "bounding_box":
            obj = a
            for i in xytheta_idx:
                obj = obj.Integrate(z[i], z_min[i], z_max[i])
            cost = 0
            for monomial,coeff in obj.monomial_to_coefficient_map().items(): 
                s1_deg = monomial.degree(z[2]) 
                c1_deg = monomial.degree(z[3])
                monomial_int1 = quad(lambda x: np.sin(x)**s1_deg * np.cos(x)**c1_deg, -np.pi/2, np.pi/2)[0]
                if np.abs(monomial_int1) <=1e-5:
                    monomial_int1 = 0
                cost += monomial_int1 * coeff
        elif integrate_region == "sublevel_set":
            cost = 0
            for monomial,coeff in a.monomial_to_coefficient_map().items(): 
                monomial_deg = []
                for i in range(nz):
                    monomial_deg.append(monomial.degree(z[i])) 
                monomial_int = integrate_subelevel_set_monte_carlo(monomial_deg, n_samples=100)
                cost += monomial_int * coeff
        
        # Make the numerics better
        prog.AddLinearCost(cost)

        # Enforce Bellman inequality.
        f_val = f(z, u_fixed)
        J_dot = J_expr.Jacobian(z).dot(f_val)
        LHS = a.ToExpression() - J_dot - l_cost(z, u_fixed)

        lam_deg = Polynomial(LHS).TotalDegree() - 2
        # S procedure for s^2 + c^2 = 1.
        lam = prog.NewFreePolynomial(Variables(z), lam_deg).ToExpression()
        S_ring = lam * (z[2]**2 + z[3]**2 - 1)

        rho_deg = Polynomial(LHS).TotalDegree() - deg
        rho_deg = int(np.ceil(rho_deg/2)*2)
        lam_rho_Jdot = prog.NewSosPolynomial(Variables(z), rho_deg)[0].ToExpression()
        S_Jdot = lam_rho_Jdot*(J_star - rho)
        if not actuator_saturate:
            prog.AddSosConstraint(LHS + S_ring + S_Jdot)

        # Enforce l(x,u)-a(x) is PD
        u = prog.NewIndeterminates(nu, 'u')
        lam_rho_la = prog.NewSosPolynomial(Variables(z), rho_deg)[0].ToExpression()
        S_la = lam_rho_la*(J_star - rho)
        lam_la = prog.NewFreePolynomial(Variables(z), lam_deg).ToExpression()
        S_ring_la = lam_la * (z[2]**2 + z[3]**2 - 1)
        prog.AddSosConstraint(l_cost(z,u) - a.ToExpression() + S_la + S_ring_la)

        # Enforce J is PD on the sublevel set
        lam_rho_J = prog.NewSosPolynomial(Variables(z), rho_deg)[0].ToExpression()
        S_J = lam_rho_J*(J_star-rho)
        lam_r = prog.NewFreePolynomial(Variables(z), deg-2).ToExpression()
        S_r = lam_r * (z[2]**2 + z[3]**2 - 1)
        prog.AddSosConstraint(J_expr + S_J + S_r)

        # Actuator saturation
        if actuator_saturate:
            LHS_limits = []
            Su_limits = []
            S_ring_limits = []
            S_Jdot_limits = []   
            for k in range(3): #(-inf, u_min),[u_min, u_max], (u_max, inf) 
                u_limit = np.zeros(nu, dtype=Expression)
                Su_limit = 0
                if k == 0:
                    u_limit[0] = u_min[0]
                elif k == 1:
                    u_limit[0] = u_fixed[0]
                elif k == 2:
                    u_limit[0] = u_max[0]
                for n in range(3):
                    if n == 0:
                        u_limit[1] = u_min[1]
                    elif n == 1:
                        u_limit[1] = u_fixed[1]
                    elif n == 2:
                        u_limit[1] = u_max[1]

                    f_limit = f(z, u_limit)
                    J_dot_limit = J_expr.Jacobian(z).dot(f_limit)
                    LHS_limit = a.ToExpression() - J_dot_limit - l_cost(z, u_limit)
                    LHS_limits.append(LHS_limit)

                    lam_u_deg = Polynomial(LHS_limit).TotalDegree() - np.max([Polynomial(u_limit[0]).TotalDegree(), Polynomial(u_limit[1]).TotalDegree()])
                    if k == 0:
                        lam_u = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                        Su_limit += lam_u*(u_fixed[0] - u_min[0])
                    elif k == 1:
                        lam_u_min = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                        lam_u_max = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                        Su_limit += lam_u_max*(u_fixed[0] - u_max[0]) + lam_u_min*(u_min[0] - u_fixed[0])
                    elif k == 2:
                        lam_u = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                        Su_limit += lam_u*(u_max[0] - u_fixed[0])

                    if n == 0:
                        lam_u = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                        Su_limit += lam_u*(u_fixed[1] - u_min[1])
                    elif n == 1:
                        lam_u_min = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                        lam_u_max = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                        Su_limit += lam_u_max*(u_fixed[1] - u_max[1]) + lam_u_min*(u_min[1] - u_fixed[1])
                    elif n == 2:
                        lam_u = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                        Su_limit += lam_u*(u_max[1] - u_fixed[1])

                    lam_limit_deg = Polynomial(LHS_limit).TotalDegree() - 2
                    lam_23 = prog.NewFreePolynomial(Variables(z), lam_limit_deg).ToExpression()
                    S_ring_limit = lam_23 * (z[2]**2 + z[3]**2 - 1)
                    S_Jdot_limit = 0
                    for i in np.arange(nz):
                        lam = prog.NewSosPolynomial(Variables(z), lam_limit_deg)[0].ToExpression()
                        S_Jdot_limit += lam*(z[i]-z_max[i])*(z[i]-z_min[i])

                    Su_limits.append(Su_limit)
                    S_ring_limits.append(S_ring_limit)
                    S_Jdot_limits.append(S_Jdot_limit)
                    prog.AddSosConstraint(LHS_limit + S_ring_limit + S_Jdot_limit + Su_limit)

        options = SolverOptions()
        options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        prog.SetSolverOptions(options)
        result = Solve(prog)
        assert result.is_success()
        a_star = result.GetSolution(a).RemoveTermsWithSmallCoefficients(1e-6).ToExpression()
        lam_rho_star = [result.GetSolution(lam_rho_Jdot).RemoveTermsWithSmallCoefficients(1e-6).ToExpression(), 
        result.GetSolution(lam_rho_J).RemoveTermsWithSmallCoefficients(1e-6).ToExpression()]

        save_polynomial(Polynomial(a_star), z, "quadrotor2d/data/iterative/iter_{}/a.pkl".format(iteration))
        save_polynomial(Polynomial(lam_rho_star[0]), z, "quadrotor2d/data/iterative/iter_{}/lam_rho_Jdot.pkl".format(iteration))
        save_polynomial(Polynomial(lam_rho_star[1]), z, "quadrotor2d/data/iterative/iter_{}/lam_rho_J.pkl".format(iteration))
        return a_star, lam_rho_star
    
    def search_upper_bound(J_old, a_star, u_fixed, rho, lam_rho_star, integrate_region="sublevel_set"):
        print("="*10, "Searching for J", "="*20)
        prog = MathematicalProgram()
        prog.AddIndeterminates(z)
        J = prog.NewFreePolynomial(Variables(z), deg)
        
        J_expr = J.ToExpression()

        def sampler():
            while True:
                x = np.random.uniform(z_min[0], z_max[0])
                y = np.random.uniform(z_min[1], z_max[1])
                theta = np.random.uniform(0, np.pi/25)
                xd = np.random.uniform(z_min[-3], z_max[-3])
                yd = np.random.uniform(z_min[-2], z_max[-2])
                thetad = np.random.uniform(z_min[-1], z_max[-1])
                z_val = x2z([x, y, theta, xd, yd, thetad])
                if J_old.Evaluate(dict(zip(z, z_val))) <= rho:
                    yield (x, y, theta, xd, yd, thetad)
        
        def integrate_subelevel_set_monte_carlo(monomial_deg, n_samples=10000):
            assert len(monomial_deg) == nz
            def integrand(x):
                assert len(x) == nx
                return x[0]**monomial_deg[0] * x[1]**monomial_deg[1] * \
                np.sin(x[2])**monomial_deg[2] * np.cos(x[2])**monomial_deg[3] * \
                    x[3]**monomial_deg[4] * x[4]**monomial_deg[5]* x[5]**monomial_deg[6]

            result, error = mcint.integrate(integrand, sampler(), measure=1, n=n_samples)
            return result 

        # Maximize volume beneath the value function.
        if objective=="integrate_all":
            obj = J
            for i in range(nz):
                obj = obj.Integrate(z[i], z_min[i], z_max[i])
            prog.AddCost(-obj.ToExpression())
        elif objective=="integrate_ring":
            if integrate_region == "sublevel_set":
                cost = 0
                for monomial,coeff in J.monomial_to_coefficient_map().items(): 
                    monomial_deg = []
                    for i in range(nz):
                        monomial_deg.append(monomial.degree(z[i])) 
                    monomial_int = integrate_subelevel_set_monte_carlo(monomial_deg, n_samples=100)
                    cost += monomial_int * coeff
            elif integrate_region == "bounding_box":
                obj = J
                for i in xytheta_idx:
                    obj = obj.Integrate(z[i], z_min[i], z_max[i])
                cost = 0
                for monomial,coeff in obj.monomial_to_coefficient_map().items(): 
                    s1_deg = monomial.degree(z[2]) 
                    c1_deg = monomial.degree(z[3])
                    monomial_int1 = quad(lambda x: np.sin(x)**s1_deg * np.cos(x)**c1_deg, -np.pi/2, np.pi/2)[0]
                    if np.abs(monomial_int1) <=1e-5:
                        monomial_int1 = 0
                    cost += monomial_int1 * coeff
            poly = Polynomial(cost)
            poly = poly.RemoveTermsWithSmallCoefficients(1e-6)
            cost_coeff = [c.Evaluate() for c in poly.monomial_to_coefficient_map().values()]
            # Make the numerics better
            prog.AddLinearCost(cost/np.max(np.abs(cost_coeff)))

        # Enforce Bellman inequality.
        f_val = f(z, u_fixed)
        J_dot = J_expr.Jacobian(z).dot(f_val)
        LHS = a_star - J_dot - l_cost(z, u_fixed)
        
        lam_deg = Polynomial(LHS).TotalDegree() - 2
        # S procedure for s^2 + c^2 = 1.
        lam = prog.NewFreePolynomial(Variables(z), lam_deg).ToExpression()
        S_ring = lam * (z[2]**2 + z[3]**2 - 1)
        S_Jdot = lam_rho_star[0]*(J_expr - rho)
        prog.AddSosConstraint(LHS + S_ring + S_Jdot)

        S_J = lam_rho_star[1]*(J_expr-rho)
        lam_r = prog.NewFreePolynomial(Variables(z), deg-2).ToExpression()
        S_r = lam_r * (z[2]**2 + z[3]**2 - 1)
        prog.AddSosConstraint(J_expr + S_J + S_r)

        # J(z0) = 0.
        J0 = J_expr.EvaluatePartial(dict(zip(z, z0)))
        prog.AddLinearConstraint(J0 == 0)

        # Solve and retrieve result.
        options = SolverOptions()
        options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        # options.SetOption(MosekSolver.id(), "writedata", "quadrotor2d_iterative_search_J.task.gz")
        prog.SetSolverOptions(options)
        result = Solve(prog)
        assert result.is_success()
        J_star = Polynomial(result.GetSolution(J_expr)).RemoveTermsWithSmallCoefficients(1e-6)

        f2_val = f2(z)
        dJdz = J_star.ToExpression().Jacobian(z)
        u_star = - .5 * Rinv.dot(f2_val.T).dot(dJdz.T)
        return J_star, u_star
    
    def maximize_rho(a_star, J_star, u_star, lam_rho_star):
        print("="*10, "Maximizing rho", "="*20)
        prog = MathematicalProgram()
        prog.AddIndeterminates(z)
        rho = prog.NewContinuousVariables(1, 'rho')[0]

        f_val = f(z, u_star)
        J_dot = J_star.Jacobian(z).dot(f_val)
        LHS = a_star - J_dot - l_cost(z, u_star)

        lam_deg = Polynomial(LHS).TotalDegree() - 2
        # S procedure for s^2 + c^2 = 1.
        lam = prog.NewFreePolynomial(Variables(z), lam_deg).ToExpression()
        S_ring = lam * (z[2]**2 + z[3]**2 - 1)
        S_Jdot = lam_rho_star[0]*(J_star - rho)
        prog.AddSosConstraint(LHS + S_ring + S_Jdot) 

        S_J = lam_rho_star[1]*(J_star - rho)
        lam_r = prog.NewFreePolynomial(Variables(z), deg-2).ToExpression()
        S_r = lam_r * (z[2]**2 + z[3]**2 - 1)
        prog.AddSosConstraint(J_star + S_J + S_r)

        prog.AddLinearCost(-rho)

        options = SolverOptions()
        options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        prog.SetSolverOptions(options)
        result = Solve(prog)
        assert result.is_success()
        
        rho_star = result.GetSolution(rho)

        print("rho: ", rho_star)

        return rho_star

    z = MakeVectorVariable(nz, "z")
    K, S = quadrotor2d_constrained_lqr()
    u_star = -K @ (z-z0) + u0
    J_star = (z-z0).dot(S).dot(z-z0) + 1e-4 * (z-z0).dot(z-z0)
    rho = 0.00039
    old_J = Polynomial(np.ones(nz)@z)

    for iteration in range(10):
        print("Iter.", iteration)
        os.makedirs("quadrotor2d/data/iterative/iter_{}".format(iteration), exist_ok=True)
        os.makedirs("quadrotor2d/figures/iterative/iter_{}".format(iteration), exist_ok=True)
        try:
            a_star = load_polynomial(z, "quadrotor2d/data/iterative/iter_{}/a.pkl".format(iteration))
            lam_rho_star = []
            lam_rho_star.append(load_polynomial(z, "quadrotor2d/data/iterative/iter_{}/lam_rho_Jdot.pkl".format(iteration)))
            lam_rho_star.append(load_polynomial(z, "quadrotor2d/data/iterative/iter_{}/lam_rho_J.pkl".format(iteration)))
        except:
            a_star, lam_rho_star = search_a(J_star, u_star, rho)
        try:
            J_star, u_star = search_upper_bound(J_star, a_star, u_star, rho, lam_rho_star)
        except:
            pass
        rho = maximize_rho(a_star, J_star, u_star, lam_rho_star)
        if Polynomial(J_star).CoefficientsAlmostEqual(old_J, 1e-3):
            print("="*10, "Converged!","="*20)
            print("Iter. ", iteration)
            break
        old_J = J_star

        if visualize:
            plot_value_function(Polynomial(J_star), z, np.array([1, 1, np.sin(np.pi/2), 1, 1, 1, 1]), u0, file_name="iterative/iter_{}/upper_bound_constrained_lqr_{}_{}".format(iteration, objective, deg), plot_states="xy", u_index=0, actuator_saturate=actuator_saturate)
    return J_star, z, z_max

if __name__ == '__main__':
    # quadrotor2d_upper_bound_ROA()
    deg = 2
    J_star, z, z_max = quadrotor2d_sos_upper_bound(deg, objective="integrate_ring",visualize=True, actuator_saturate=False)

    C = extract_polynomial_coeff_dict(J_star, z)
    f = open("quadrotor2d/data/{}/J_upperer_bound_deg_{}.pkl".format(z_max, deg),"wb")
    pickle.dump(C, f)
    f.close()
