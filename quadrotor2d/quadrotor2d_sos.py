import sys
sys.path.append("../underactuated")

import numpy as np
from scipy.integrate import quad
from utils import extract_polynomial_coeff_dict, calc_u_opt 
import pickle
from pydrake.all import (MathematicalProgram, Variables, Expression, Solve, Polynomial, SolverOptions, CommonSolverOption, 
Linearize, LinearQuadraticRegulator, DiagramBuilder, AddMultibodyPlantSceneGraph, Parser)
from underactuated.quadrotor2d import Quadrotor2D

import matplotlib.pyplot as plt
from matplotlib import cm

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
    z_max = np.array([1, 1, 1, 1, 4.5, 4.5, 3])
    z_min = np.array([-1, -1, -1, -1, -4.5, -4.5, -3])

    # Equilibrium point in both the system coordinates.
    x0 = np.zeros(nx)
    z0 = x2z(x0)
    z0[np.abs(z0)<=1e-6] = 0
        
    # Quadratic running cost in augmented state.
    Q = np.diag([10, 10, 10, 10, 1, 1, r/(2*np.pi)]) * 1e2
    R = np.array([[0.1, 0.05], [0.05, 0.1]])
    def l_cost(z, u):
        return (z - z0).dot(Q).dot(z - z0) + u.dot(R).dot(u)

    Rinv = np.linalg.inv(R)

    if test:
        return nz, f, f2, x2z, Rinv

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
        # prog.AddLinearCost(-c_r * cost)
    # prog.AddQuadraticCost(1e-3*np.sum(np.array(list(J.decision_variables()))**2), is_convex=True)

    # S procedure for s^2 + c^2 = 1.
    lam = prog.NewFreePolynomial(Variables(z), deg).ToExpression()
    S_ring = lam * (z[2]**2 + z[3]**2 - 1)
    
    # Enforce Bellman inequality.
    f_val = f(z, u)
    J_dot = J_expr.Jacobian(z).dot(f_val)
    if deg <= 0:
        S_Jdot = 0
        for i in xytheta_idx:
            lam = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
            S_Jdot += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
        prog.AddSosConstraint(J_dot + l_cost(z, u) + S_ring + S_Jdot)
    else:
        prog.AddSosConstraint(J_dot + l_cost(z, u) + S_ring)

    # Enforce that value function is PD
    if deg <= 0:
        S_J = 0
        for i in xytheta_idx:
            lam = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
            S_J += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
        prog.AddSosConstraint(J_expr + S_J)
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

    if visualize:
        plot_value_function(J_star, z, z_max, u0, file_name="lower_bound_{}_{}".format(objective, deg), plot_states="xy", u_index=0)
    return J_star, z

def plot_value_function(J_star, z, z_max, u0, file_name="", plot_states="xy", u_index=0):
    nz = 7
    x_max = np.zeros(6)
    x_max[:2] = z_max[:2]
    x_max[2] = np.pi/2
    x_max[3:] = z_max[4:]
    x_min = -x_max

    dJdz = J_star.ToExpression().Jacobian(z)

    nz, f, f2, x2z, Rinv = quadrotor2d_sos_lower_bound(2, test=True)

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


if __name__ == '__main__':
    deg = 2
    J_star, z = quadrotor2d_sos_lower_bound(deg, visualize=True)

    C = extract_polynomial_coeff_dict(J_star, z)
    f = open("quadrotor2d/data/J_lower_bound_deg_{}.pkl".format(deg),"wb")
    pickle.dump(C, f)
    f.close()
