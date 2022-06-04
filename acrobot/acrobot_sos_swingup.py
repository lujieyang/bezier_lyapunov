import numpy as np
from scipy.integrate import quad
from utils import extract_polynomial_coeff_dict
import pickle
from pydrake.all import (MathematicalProgram, Variables, Expression, Solve, Polynomial, SolverOptions, CommonSolverOption)
from pydrake.examples.acrobot import AcrobotParams
from acrobot_swingup_fvi import plot_value_function, acrobot_setup

# Given the degree for the approximate value function and the polynomials
# in the S procedure, solves the SOS and returns the approximate value function
# (together with the objective of the SOS program).
def acrobot_sos_lower_bound(deg, objective="integrate_ring", visualize=False, test=False):
    nz = 6
    nq = 2
    nx = 2 * nq
    nu = 1

    params = AcrobotParams()
    m1 = params.m1()
    m2 = params.m2()
    l1 = params.l1()
    lc1 = params.lc1()
    lc2 = params.lc2()
    I1 = params.Ic1() + m1*lc1**2
    I2 = params.Ic2() + m2*lc2**2
    g = params.gravity()
    B = np.array([[0], [1]])
    # Map from original state to augmented state.
    # Uses sympy to be able to do symbolic integration later on.
    # x = (theta1, theta2, theta1dot, theta2dot)
    # z = (s1, c1, s2, c2, theta1dot, theta2dot)
    x2z = lambda x : np.array([np.sin(x[0]), np.cos(x[0]), np.sin(x[1]), np.cos(x[1]), x[2], x[3]])


    def T(z, dtype=Expression):
        assert len(z) == nz
        T = np.zeros([nz, nx], dtype=dtype)
        T[0, 0] = z[1]
        T[1, 0] = -z[0]
        T[2, 1] = z[3]
        T[3, 1] = -z[2]
        T[4, 2] = 1
        T[5, 3] = 1
        return T

    def M0(z, dtype=Expression):
        assert len(z) == nz
        s2 = z[2]
        c2 = z[3]
        qdot = z[4:]
        M = np.array([[I1 + I2 + m2*l1**2 + 2*m2*l1*lc2*c2,  I2 + m2*l1*lc2*c2],
        [I2 + m2*l1*lc2*c2, I2]])
        entry = m2*l1*lc2*s2*qdot[1]
        C = np.array([[-2*entry, -entry],
        [m2*l1*lc2*s2*qdot[0], 0]])
        M0_val = np.zeros([nx, nx], dtype=dtype)
        M0_val[:nq, :nq] = np.eye(nq)
        M0_val[nq:, :nq] = C
        M0_val[nq:, nq:] = M
        return M0_val

    def F(z, u, dtype=Expression):
        assert len(z) == nz
        s1 = z[0]
        c1 = z[1]
        s2 = z[2]
        c2 = z[3]
        # s12 = sin(theta1 + theta2) = s1c2 + c1s2
        s12 = s1*c2 + c1*s2
        F_val = np.zeros(nx, dtype=dtype)
        F_val[:nq] = z[4:]
        tau_q = np.array([-m1*g*lc1*s1 - m2*g*(l1*s1+lc2*s12),-m2*g*lc2*s12])
        F_val[nq:] = tau_q + B @ u
        return F_val

    def f2(z, T, dtype=Expression):
        assert len(z) == nz
        Minv = np.linalg.inv(M(z))
        f2_val = np.zeros([nx, nu], dtype=dtype)
        f2_val[nq:, :] = Minv @ B
        return T@f2_val

    def M(z):
        assert len(z) == nz
        c2 = z[3]
        M_val = np.array([[I1 + I2 + m2*l1**2 + 2*m2*l1*lc2*c2,  I2 + m2*l1*lc2*c2],
        [I2 + m2*l1*lc2*c2, I2]])
        return M_val

    if test:
        return M0, F, M
    
    # State limits (region of state space where we approximate the value function).
    z_max = np.array([1, 1, 1, 1, 3, 3])
    z_min = np.array([-1, -1, -1, 0, -3, -3])

    # Equilibrium point in both the system coordinates.
    x0 = np.array([np.pi, 0, 0, 0])
    z0 = x2z(x0)
    z0[np.abs(z0)<=1e-6] = 0
        
    # Quadratic running cost in augmented state.
    # Q = np.diag([1e5, 1e5, 1e5, 1e5, 5e4, 5e4])
    Q = np.diag([1000, 1000, 1000, 1000, 500, 500])
    R = np.diag([1]) 
    def l_cost(z, u):
        return (z - z0).dot(Q).dot(z - z0) + u.dot(R).dot(u)

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
        for i in range(2*nq, nz):
            obj = obj.Integrate(z[i], z_min[i], z_max[i])
        c_r = 1
        cost = 0
        for m,coeff in obj.monomial_to_coefficient_map().items(): 
            s1_deg = m.degree(z[0]) 
            c1_deg = m.degree(z[1])
            s2_deg = m.degree(z[2]) 
            c2_deg = m.degree(z[3])
            monomial_int1 = quad(lambda x: np.sin(x)**s1_deg * np.cos(x)**c1_deg, 0, 2*np.pi)[0]
            monomial_int2 = quad(lambda x: np.sin(x)**s2_deg * np.cos(x)**c2_deg, 0, 2*np.pi)[0]
            if np.abs(monomial_int1) <=1e-5:
                monomial_int1 = 0
            if np.abs(monomial_int2) <=1e-5:
                monomial_int2 = 0
            cost += monomial_int1 * monomial_int2 * coeff
        poly = Polynomial(cost)
        cost_coeff = [c.Evaluate() for c in poly.monomial_to_coefficient_map().values()]
        # Make the numerics better
        prog.AddLinearCost(-c_r * cost/np.max(np.abs(cost_coeff)))
        # prog.AddLinearCost(-c_r * cost)
    # Regularization on J coefficients
    # prog.AddQuadraticCost(1e-5*np.sum(np.array(list(J.decision_variables()))**2), is_convex=True)

    # S procedure for s^2 + c^2 = 1.
    lam01 = prog.NewFreePolynomial(Variables(z), deg).ToExpression()
    S_ring = lam01 * (z[0]**2 + z[1]**2 - 1)
    lam23 = prog.NewFreePolynomial(Variables(z), deg).ToExpression()
    S_ring += lam23 * (z[2]**2 + z[3]**2 - 1)

    # Enforce Bellman inequality.
    T_val = T(z)
    M0_val = M0(z)
    F_val = F(z, u)
    w = prog.NewIndeterminates(nx, 'w')
    J_dot = J_expr.Jacobian(z).dot(T_val@w)
    dynamics = M0_val@w - F_val
    S_w = 0
    for i in range(len(dynamics)):
        lamw = prog.NewFreePolynomial(Variables(np.concatenate((z, w, u))), deg).ToExpression()
        S_w += lamw * dynamics[i]
    if deg <= 0:
        # S procedure for compact domain 
        lam_Jdot_4 = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
        S_Jdot = lam_Jdot_4 * (z[4]-z_max[4]) * (z[4]-z_min[4])
        lam_Jdot_5 = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
        S_Jdot += lam_Jdot_5 * (z[5]-z_max[5]) * (z[5]-z_min[5])
        prog.AddSosConstraint(J_dot + l_cost(z, u) + S_ring + S_Jdot + S_w)
    else:
        prog.AddSosConstraint(J_dot + l_cost(z, u) + S_ring + S_w)

    # Enforce that value function is PD
    if deg <= 0:
        lam_J_4 = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
        S_J = lam_J_4 * (z[4]-z_max[4]) * (z[4]-z_min[4])
        lam_J_5 = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
        S_J += lam_J_5 * (z[5]-z_max[5]) * (z[5]-z_min[5])
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
        params_dict = acrobot_setup()
        params_dict["x_max"] = np.array([2*np.pi, np.pi/2, 3, 3])
        params_dict["x_min"] = np.array([0, -np.pi/2, -3, -3])
        plot_value_function(J_star, z, params_dict, deg, file_name="sos/lower_bound_{}".format(objective))
    return J_star, z

if __name__ == '__main__':
    deg = 6
    J_star, z = acrobot_sos_lower_bound(deg, visualize=True)

    C = extract_polynomial_coeff_dict(J_star, z)
    f = open("acrobot/data/sos/J_lower_bound_deg_{}.pkl".format(deg),"wb")
    pickle.dump(C, f)
    f.close()
