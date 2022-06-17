import numpy as np
from scipy.integrate import quad
from torch import adjoint
from utils import extract_polynomial_coeff_dict, matrix_adjoint, matrix_det
import pickle
from pydrake.all import (MathematicalProgram, Variables, Expression, Solve, Polynomial, SolverOptions, CommonSolverOption, 
LinearQuadraticRegulator, MakeVectorVariable)
from pydrake.examples.acrobot import AcrobotParams
from acrobot_swingup_fvi import plot_value_function, acrobot_setup

# Given the degree for the approximate value function and the polynomials
# in the S procedure, solves the SOS and returns the approximate value function
# (together with the objective of the SOS program).
def acrobot_sos_lower_bound_implicit_dynamics(deg, objective="integrate_ring", visualize=False, test=False):
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
    z_max = np.array([1, 1, 1, 1, 1, 1])
    z_min = np.array([-1, -1, -1, 0, -1, -1]) 

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

    non_sc_idx = [4, 5]

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
        for i in non_sc_idx:
            obj = obj.Integrate(z[i], z_min[i], z_max[i])
        c_r = 1
        cost = 0
        for m,coeff in obj.monomial_to_coefficient_map().items(): 
            s1_deg = m.degree(z[0]) 
            c1_deg = m.degree(z[1])
            s2_deg = m.degree(z[2]) 
            c2_deg = m.degree(z[3])
            monomial_int1 = quad(lambda x: np.sin(x)**s1_deg * np.cos(x)**c1_deg, 0, 2*np.pi)[0]
            monomial_int2 = quad(lambda x: np.sin(x)**s2_deg * np.cos(x)**c2_deg, 0, np.pi)[0]
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
        params_dict["x_max"] = np.array([2*np.pi, np.pi/2, 1, 1])
        params_dict["x_min"] = np.array([0, -np.pi/2, -1, -1])
        plot_value_function(J_star, z, params_dict, deg, file_name="sos/lower_bound_{}".format(objective))
    return J_star, z

def acrobot_sos_lower_bound_explicit_dynamics(deg, objective="integrate_ring", visualize=False, test=False):
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

    def f(z, u, dtype=Expression):
        assert len(z) == nz
        s1 = z[0]
        c1 = z[1]
        s2 = z[2]
        c2 = z[3]
        # s12 = sin(theta1 + theta2) = s1c2 + c1s2
        s12 = s1*c2 + c1*s2
        qdot = z[4:]
        M = np.array([[I1 + I2 + m2*l1**2 + 2*m2*l1*lc2*c2,  I2 + m2*l1*lc2*c2],
        [I2 + m2*l1*lc2*c2, I2]])
        det_M = matrix_det(M)
        f_val = np.zeros(nz, dtype=dtype)
        f_val[0] = qdot[0] * c1 * det_M
        f_val[1] = -qdot[0] * s1 * det_M
        f_val[2] = qdot[1] * c2 * det_M
        f_val[3] = -qdot[1] * s2 * det_M
        tau_q = np.array([-m1*g*lc1*s1 - m2*g*(l1*s1+lc2*s12),-m2*g*lc2*s12])
        entry = m2*l1*lc2*s2*qdot[1]
        C = np.array([[-2*entry, -entry],
        [m2*l1*lc2*s2*qdot[0], 0]])
        f_val[4:] = matrix_adjoint(M) @ (tau_q - C@qdot + B @ u)
        return f_val, det_M

    if test:
         return f, T

    # State limits (region of state space where we approximate the value function).
    dz = 0.01
    z_max = np.array([np.sin(np.pi-dz), np.cos(np.pi-dz), np.sin(dz), 1, dz, dz])
    z_min = np.array([-np.sin(np.pi-dz), -1, -np.sin(dz), np.cos(dz), -dz, -dz])

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

    Rinv = np.linalg.inv(R)

    non_sc_idx = [4, 5]

    # Set up optimization.        
    prog = MathematicalProgram()
    z = prog.NewIndeterminates(nz, 'z')
    u = prog.NewIndeterminates(nu, 'u')
    J = prog.NewFreePolynomial(Variables(z), deg)
    J_expr = J.ToExpression()

    # S procedure for s^2 + c^2 = 1.
    lam_01 = prog.NewFreePolynomial(Variables(z), deg).ToExpression()
    lam_23 = prog.NewFreePolynomial(Variables(z), deg).ToExpression()
    S_ring = lam_01 * (z[0]**2 + z[1]**2 - 1) + lam_23 * (z[2]**2 + z[3]**2 - 1)

    # Enforce Bellman inequality.
    f_val, det_M = f(z, u)
    J_dot = J_expr.Jacobian(z).dot(f_val)
    if deg >= 0:
        S_Jdot = 0
        # Also constrain theta to be in [-pi/2, pi/2]
        for i in np.arange(nz):
            lam = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
            S_Jdot += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
        prog.AddSosConstraint(J_dot + l_cost(z, u) * det_M + S_ring + S_Jdot)
    else:
        prog.AddSosConstraint(J_dot + l_cost(z, u) * det_M + S_ring)

    # Enforce that value function is PD
    if deg >= 0:
        S_J = 0
        # Also constrain theta to be in [-pi/2, pi/2]
        for i in np.arange(nz):
            lam = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
            S_J += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
        prog.AddSosConstraint(J_expr + S_J)
    else:
        prog.AddSosConstraint(J_expr)

    # J(z0) = 0.
    J0 = J_expr.EvaluatePartial(dict(zip(z, z0)))
    prog.AddLinearConstraint(J0 == 0)

    # Maximize volume beneath the value function.
    if objective=="integrate_all":
        obj = J
        for i in range(nz):
            obj = obj.Integrate(z[i], z_min[i], z_max[i])
        prog.AddCost(-obj.ToExpression())
    elif objective=="integrate_ring":
        obj = J
        for i in non_sc_idx:
            obj = obj.Integrate(z[i], z_min[i], z_max[i])
        c_r = 1
        cost = 0
        for m,coeff in obj.monomial_to_coefficient_map().items(): 
            s1_deg = m.degree(z[0]) 
            c1_deg = m.degree(z[1])
            s2_deg = m.degree(z[2]) 
            c2_deg = m.degree(z[3])
            monomial_int1 = quad(lambda x: np.sin(x)**s1_deg * np.cos(x)**c1_deg, np.pi-dz, np.pi+dz)[0]
            monomial_int2 = quad(lambda x: np.sin(x)**s2_deg * np.cos(x)**c2_deg, -dz, dz)[0]
            cost += monomial_int1 * monomial_int2 * coeff
        poly = Polynomial(cost)
        cost_coeff = [c.Evaluate() for c in poly.monomial_to_coefficient_map().values()]
        # Make the numerics better
        prog.AddLinearCost(-c_r * cost/np.max(np.abs(cost_coeff)))

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    assert result.is_success()
    J_star = Polynomial(result.GetSolution(J_expr)).RemoveTermsWithSmallCoefficients(1e-6)
    l_val = Polynomial(result.GetSolution(l_cost(z, u)))

    if visualize:
        params_dict = acrobot_setup()
        params_dict["x_max"] = np.array([np.pi+dz, dz, z_max[-2], z_max[-1]])
        params_dict["x_min"] = np.array([np.pi-dz, -dz,  z_min[-2], z_min[-1]])
        plot_value_function(J_star, z, params_dict, deg, file_name="sos/lower_bound_explicit_dynamics_{}".format(objective))
    return J_star, z

def acrobot_constrained_lqr(nz=6, nu=1):
    params = AcrobotParams()
    m1 = params.m1()
    m2 = params.m2()
    l1 = params.l1()
    lc1 = params.lc1()
    lc2 = params.lc2()
    I1 = params.Ic1() + m1*lc1**2
    I2 = params.Ic2() + m2*lc2**2
    g = params.gravity()
    A = np.zeros([nz, nz])
    B = np.zeros([nz, nu])
    A[0, 4] = -1
    A[2, 5] = 1
    A[4, 0] = -g*(I2*lc1*m1 + I2*l1*m2 - l1*lc2**2*m2**2)/(I1*I2 + l1**2*m2*(I2 - lc2**2*m2))
    A[5, 0] = g*(I2*(lc1*m1 + l1*m2) - lc2*m2*(I1 - l1*lc1*m1 + l1*lc2*m2))/(I1*I2 + l1**2*m2*(I2 - lc2**2*m2))
    A[4, 2] = -(g*l1*lc2**2*m2**2)/(I1*I2 + l1**2*m2*(I2 - lc2**2*m2))
    A[5, 2] = g*lc2*m2*(I1 + l1*(l1 + lc2)*m2)/(I1*I2 + l1**2*m2*(I2 - lc2**2*m2))
    B[4, :] = -((I2 + l1*lc2*m2)/(I1*I2 + l1**2*m2*(I2 - lc2**2*m2)))
    B[5, :] = (I1 + I2 + l1*(l1 + 2*lc2)*m2)/(I1*I2 + l1**2*m2*(I2 - lc2**2*m2))
    F = np.zeros([2, nz])
    F[0, 1] = -1
    F[1, 3] = 1
    Q = np.diag([100, 100, 100, 100, 1, 1])
    R = np.array([1])
    K, S = LinearQuadraticRegulator(A, B, Q, R, F=F)
    return K

def acrobot_sos_upper_bound(deg, deg_lower=0, objective="integrate_ring", visualize=False, test=False):
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

    def f(z, u, dtype=Expression):
        assert len(z) == nz
        s1 = z[0]
        c1 = z[1]
        s2 = z[2]
        c2 = z[3]
        # s12 = sin(theta1 + theta2) = s1c2 + c1s2
        s12 = s1*c2 + c1*s2
        qdot = z[4:]
        M = np.array([[I1 + I2 + m2*l1**2 + 2*m2*l1*lc2*c2,  I2 + m2*l1*lc2*c2],
        [I2 + m2*l1*lc2*c2, I2]])
        det_M = matrix_det(M)
        f_val = np.zeros(nz, dtype=dtype)
        f_val[0] = qdot[0] * c1 * det_M
        f_val[1] = -qdot[0] * s1 * det_M
        f_val[2] = qdot[1] * c2 * det_M
        f_val[3] = -qdot[1] * s2 * det_M
        tau_q = np.array([-m1*g*lc1*s1 - m2*g*(l1*s1+lc2*s12),-m2*g*lc2*s12])
        entry = m2*l1*lc2*s2*qdot[1]
        C = np.array([[-2*entry, -entry],
        [m2*l1*lc2*s2*qdot[0], 0]])
        f_val[4:] = matrix_adjoint(M) @ (tau_q - C@qdot + B @ u)
        return f_val, det_M

    # def M(z):
    #     assert len(z) == nz
    #     c2 = z[3]
    #     M_val = np.array([[I1 + I2 + m2*l1**2 + 2*m2*l1*lc2*c2,  I2 + m2*l1*lc2*c2],
    #     [I2 + m2*l1*lc2*c2, I2]])
    #     return M_val

    if test:
         return f, T

    # State limits (region of state space where we approximate the value function).
    dz = 0.01
    z_max = np.array([np.sin(np.pi-dz), np.cos(np.pi-dz), np.sin(dz), 1, dz, dz])
    z_min = np.array([-np.sin(np.pi-dz), -1, -np.sin(dz), np.cos(dz), -dz, -dz])

    # Equilibrium point in both the system coordinates.
    x0 = np.array([np.pi, 0, 0, 0])
    z0 = x2z(x0)
    z0[np.abs(z0)<=1e-6] = 0
        
    # Quadratic running cost in augmented state.
    # Q = np.diag([1e5, 1e5, 1e5, 1e5, 5e4, 5e4])
    Q = np.diag([100, 100, 100, 100, 1, 1])
    R = np.diag([1]) 
    def l_cost(z, u):
        return (z - z0).dot(Q).dot(z - z0) + u.dot(R).dot(u)

    Rinv = np.linalg.inv(R)

    non_sc_idx = [4, 5]

    # Set up optimization.        
    prog = MathematicalProgram()
    z = prog.NewIndeterminates(nz, 'z')
    K = acrobot_constrained_lqr()
    u_fixed = -K @ (z-z0)
    J = prog.NewFreePolynomial(Variables(z), deg)
    J_expr = J.ToExpression()

    a = prog.NewSosPolynomial(Variables(z), deg)[0]

    # Minimize volume beneath the a(x).
    obj = a
    for i in non_sc_idx:
        obj = obj.Integrate(z[i], z_min[i], z_max[i])
    c_r = 1
    cost = 0
    for m,coeff in obj.monomial_to_coefficient_map().items(): 
        s1_deg = m.degree(z[0]) 
        c1_deg = m.degree(z[1])
        s2_deg = m.degree(z[2]) 
        c2_deg = m.degree(z[3])
        monomial_int1 = quad(lambda x: np.sin(x)**s1_deg * np.cos(x)**c1_deg, np.pi-dz, np.pi+dz)[0]
        monomial_int2 = quad(lambda x: np.sin(x)**s2_deg * np.cos(x)**c2_deg, -dz, dz)[0]
        cost += monomial_int1 * monomial_int2 * coeff
    poly = Polynomial(cost)
    cost_coeff = [c.Evaluate() for c in poly.monomial_to_coefficient_map().values()]
    # Make the numerics better
    a_cost = prog.AddLinearCost(c_r * cost/np.max(np.abs(cost_coeff)))

    # Enforce Bellman inequality.
    f_val, det_M = f(z, u_fixed)
    J_dot = J_expr.Jacobian(z).dot(f_val)
    LHS = a.ToExpression() * det_M - J_dot - l_cost(z, u_fixed) * det_M

    lam_deg = Polynomial(LHS).TotalDegree()-2

    # S procedure for s^2 + c^2 = 1.
    lam_01 = prog.NewFreePolynomial(Variables(z), lam_deg).ToExpression()
    lam_23 = prog.NewFreePolynomial(Variables(z), lam_deg).ToExpression()
    S_ring = lam_01 * (z[0]**2 + z[1]**2 - 1) + lam_23 * (z[2]**2 + z[3]**2 - 1)

    if deg >= 0:
        S_Jdot = 0
        # Also constrain theta to be in [-pi/2, pi/2]
        for i in np.arange(nz):
            lam = prog.NewSosPolynomial(Variables(z), lam_deg)[0].ToExpression()
            S_Jdot += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
        prog.AddSosConstraint(LHS + S_ring + S_Jdot)
    else:
        prog.AddSosConstraint(LHS + S_ring)

    # Enforce that value function is PD
    if deg >= 0:
        S_J = 0
        # Also constrain theta to be in [-pi/2, pi/2]
        for i in np.arange(nz):
            lam = prog.NewSosPolynomial(Variables(z), lam_deg)[0].ToExpression()
            S_J += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
        prog.AddSosConstraint(J_expr + S_J)
    else:
        prog.AddSosConstraint(J_expr)

    # Enforce l(x,u)-a(x) is PD
    u = prog.NewIndeterminates(nu, 'u')
    if deg >= 0:
        S_la = 0
        # Also constrain theta to be in [-pi/2, pi/2]
        for i in np.arange(nz):
            lam = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
            S_la += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
        prog.AddSosConstraint(l_cost(z,u) - a.ToExpression() + S_la)
    else:
        prog.AddSosConstraint(l_cost(z,u) - a.ToExpression())

    # J(z0) = 0.
    J0 = J_expr.EvaluatePartial(dict(zip(z, z0)))
    prog.AddLinearConstraint(J0 == 0)

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    assert result.is_success()
    a_star = result.GetSolution(a).RemoveTermsWithSmallCoefficients(1e-6).ToExpression()

    prog.RemoveCost(a_cost)

    # Maximize volume beneath the value function.
    if objective=="integrate_all":
        obj = J
        for i in range(nz):
            obj = obj.Integrate(z[i], z_min[i], z_max[i])
        prog.AddCost(-obj.ToExpression())
    elif objective=="integrate_ring":
        obj = J
        for i in non_sc_idx:
            obj = obj.Integrate(z[i], z_min[i], z_max[i])
        c_r = 1
        cost = 0
        for m,coeff in obj.monomial_to_coefficient_map().items(): 
            s1_deg = m.degree(z[0]) 
            c1_deg = m.degree(z[1])
            s2_deg = m.degree(z[2]) 
            c2_deg = m.degree(z[3])
            monomial_int1 = quad(lambda x: np.sin(x)**s1_deg * np.cos(x)**c1_deg, np.pi-dz, np.pi+dz)[0]
            monomial_int2 = quad(lambda x: np.sin(x)**s2_deg * np.cos(x)**c2_deg, -dz, dz)[0]
            cost += monomial_int1 * monomial_int2 * coeff
        poly = Polynomial(cost)
        cost_coeff = [c.Evaluate() for c in poly.monomial_to_coefficient_map().values()]
        # Make the numerics better
        prog.AddLinearCost(c_r * cost/np.max(np.abs(cost_coeff)))

    # Enforce Bellman inequality.
    if deg >= 0:
        prog.AddSosConstraint(a_star * det_M - J_dot - l_cost(z, u_fixed) * det_M + S_ring + S_Jdot)
    else:
        prog.AddSosConstraint(a_star * det_M - J_dot - l_cost(z, u_fixed) * det_M + S_ring)

    result = Solve(prog)
    assert result.is_success()
    J_star = Polynomial(result.GetSolution(J_expr)).RemoveTermsWithSmallCoefficients(1e-6)
    l_val = Polynomial(result.GetSolution(l_cost(z, u_fixed)))

    # dJdz = J_star.ToExpression().Jacobian(z)
    # u_star = - .5 * Rinv.dot(f2(z).T).dot(dJdz.T)
    # xdot = f(z, u_star)
    # Jdot = dJdz.dot(xdot)
    # prog1 = MathematicalProgram()
    # prog1.AddIndeterminates(z)
    # lam = prog1.NewFreePolynomial(Variables(z), deg).ToExpression()
    # S_ring = lam * (z[2]**2 + z[3]**2 - 1)
    # S_Jdot = 0
    # # Also constrain theta to be in [-pi/2, pi/2]
    # for i in np.arange(nz):
    #     lam = prog1.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
    #     S_Jdot += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
    # prog1.AddSosConstraint(-Jdot + S_Jdot + S_ring)
    # options = SolverOptions()
    # options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    # prog1.SetSolverOptions(options)
    # result1 = Solve(prog1)
    # assert result1.is_success()

    if visualize:
        params_dict = acrobot_setup()
        params_dict["x_max"] = np.array([np.pi+dz, dz, z_max[-2], z_max[-1]])
        params_dict["x_min"] = np.array([np.pi-dz, -dz,  z_min[-2], z_min[-1]])
        plot_value_function(J_star, z, params_dict, deg, file_name="sos/upper_bound_constrained_LQR_{}".format(objective))
    return J_star, z

def acrobot_sos_iterative_upper_bound(deg, deg_lower=0, objective="integrate_ring", visualize=False, test=False, z_incre=0.01, actuator_saturate=True):
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
    u_max = np.array([1000])
    u_min = -u_max
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

    def f(z, u, dtype=Expression, u_denominator=1):
        assert len(z) == nz
        s1 = z[0]
        c1 = z[1]
        s2 = z[2]
        c2 = z[3]
        # s12 = sin(theta1 + theta2) = s1c2 + c1s2
        s12 = s1*c2 + c1*s2
        qdot = z[4:]
        M = np.array([[I1 + I2 + m2*l1**2 + 2*m2*l1*lc2*c2,  I2 + m2*l1*lc2*c2],
        [I2 + m2*l1*lc2*c2, I2]])
        det_M = matrix_det(M)
        f_val = np.zeros(nz, dtype=dtype)
        f_val[0] = qdot[0] * c1 * det_M * u_denominator
        f_val[1] = -qdot[0] * s1 * det_M * u_denominator
        f_val[2] = qdot[1] * c2 * det_M * u_denominator
        f_val[3] = -qdot[1] * s2 * det_M * u_denominator
        tau_q = np.array([-m1*g*lc1*s1 - m2*g*(l1*s1+lc2*s12),-m2*g*lc2*s12])
        entry = m2*l1*lc2*s2*qdot[1]
        C = np.array([[-2*entry, -entry],
        [m2*l1*lc2*s2*qdot[0], 0]])
        f_val[4:] = matrix_adjoint(M) @ (tau_q * u_denominator - C@qdot * u_denominator + B @ u)
        return f_val, det_M

    def f2(z, dtype=Expression):
        assert len(z) == nz
        s1 = z[0]
        c1 = z[1]
        s2 = z[2]
        c2 = z[3]
        # s12 = sin(theta1 + theta2) = s1c2 + c1s2
        s12 = s1*c2 + c1*s2
        M = np.array([[I1 + I2 + m2*l1**2 + 2*m2*l1*lc2*c2,  I2 + m2*l1*lc2*c2],
        [I2 + m2*l1*lc2*c2, I2]])
        f2_val = np.zeros([nz, nu], dtype=dtype)
        f2_val[4:, :] = matrix_adjoint(M) @  B
        return f2_val

    if test:
         return nz, f, f2, T

    # Equilibrium point in both the system coordinates.
    x0 = np.array([np.pi, 0, 0, 0])
    z0 = x2z(x0)
    z0[np.abs(z0)<=1e-6] = 0
        
    # Quadratic running cost in augmented state.
    Q = np.diag([100, 100, 100, 100, 1, 1])
    R = np.diag([1]) 
    def l_cost(z, u):
        return (z - z0).dot(Q).dot(z - z0) + u.dot(R).dot(u)

    Rinv = np.linalg.inv(R)

    non_sc_idx = [4, 5]

    def search_upper_bound(u_fixed, dz, u_denominator=1):   
        # State limits (region of state space where we approximate the value function).
        z_max = np.array([np.sin(np.pi-dz), np.cos(np.pi-dz), np.sin(dz), 1, dz, dz])
        z_min = np.array([-np.sin(np.pi-dz), -1, -np.sin(dz), np.cos(dz), -dz, -dz])   
        prog = MathematicalProgram()
        prog.AddIndeterminates(z)
        J = prog.NewFreePolynomial(Variables(z), deg)
        J_expr = J.ToExpression()

        a = prog.NewSosPolynomial(Variables(z), deg)[0]

        # Minimize volume beneath the a(x).
        obj = a
        for i in non_sc_idx:
            obj = obj.Integrate(z[i], z_min[i], z_max[i])
        c_r = 1
        cost = 0
        for m,coeff in obj.monomial_to_coefficient_map().items(): 
            s1_deg = m.degree(z[0]) 
            c1_deg = m.degree(z[1])
            s2_deg = m.degree(z[2]) 
            c2_deg = m.degree(z[3])
            monomial_int1 = quad(lambda x: np.sin(x)**s1_deg * np.cos(x)**c1_deg, np.pi-dz, np.pi+dz)[0]
            monomial_int2 = quad(lambda x: np.sin(x)**s2_deg * np.cos(x)**c2_deg, -dz, dz)[0]
            cost += monomial_int1 * monomial_int2 * coeff
        poly = Polynomial(cost)
        cost_coeff = [c.Evaluate() for c in poly.monomial_to_coefficient_map().values()]
        # Make the numerics better
        cost = cost/np.max(np.abs(cost_coeff))
        a_cost = prog.AddLinearCost(c_r * cost)

        # Enforce Bellman inequality.
        f_val, det_M = f(z, u_fixed, u_denominator=u_denominator)
        J_dot = J_expr.Jacobian(z).dot(f_val)
        LHS = a.ToExpression() * det_M * u_denominator - J_dot - l_cost(z, u_fixed) * det_M * u_denominator

        # Need higher degree lambda's for polynomial u_denominator
        lam_deg = Polynomial(LHS).TotalDegree() - 2

        # S procedure for s^2 + c^2 = 1.
        lam_01 = prog.NewFreePolynomial(Variables(z), lam_deg).ToExpression()
        lam_23 = prog.NewFreePolynomial(Variables(z), lam_deg).ToExpression()
        S_ring = lam_01 * (z[0]**2 + z[1]**2 - 1) + lam_23 * (z[2]**2 + z[3]**2 - 1)

        if deg >= 0:
            S_Jdot = 0
            # Also constrain theta to be in [-pi/2, pi/2]
            for i in np.arange(nz):
                lam = prog.NewSosPolynomial(Variables(z), lam_deg)[0].ToExpression()
                S_Jdot += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
            if actuator_saturate:
                lam_u_deg = Polynomial(LHS).TotalDegree() - Polynomial(u_fixed[0]).TotalDegree()**2
                lam_u = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                Su = lam_u*(u_fixed[0]**2 - u_max[0]**2)
                prog.AddSosConstraint(LHS + S_ring + S_Jdot + Su)
            else:
                prog.AddSosConstraint(LHS + S_ring + S_Jdot)
        else:
            prog.AddSosConstraint(LHS + S_ring)

        # Enforce that value function is PD
        if deg >= 0:
            S_J = 0
            # Also constrain theta to be in [-pi/2, pi/2]
            for i in np.arange(nz):
                lam = prog.NewSosPolynomial(Variables(z), lam_deg)[0].ToExpression()
                S_J += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
            prog.AddSosConstraint(J_expr + S_J)
        else:
            prog.AddSosConstraint(J_expr)

        # Enforce l(x,u)-a(x) is PD
        u = prog.NewIndeterminates(nu, 'u')
        if deg >= 0:
            S_la = 0
            # Also constrain theta to be in [-pi/2, pi/2]
            for i in np.arange(nz):
                lam = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
                S_la += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
            prog.AddSosConstraint(l_cost(z,u) - a.ToExpression() + S_la)
        else:
            prog.AddSosConstraint(l_cost(z,u) - a.ToExpression())

        # J(z0) = 0.
        J0 = J_expr.EvaluatePartial(dict(zip(z, z0)))
        prog.AddLinearConstraint(J0 == 0)

        # Actuator saturation
        LHS_limits = []
        Sus = []
        S_ring_limits = []
        S_Jdot_limits = []
        if actuator_saturate:
            for u_limit in [u_max, u_min]:
                f_limit, _ = f(z, u_limit, u_denominator=1)
                J_dot_limit = J_expr.Jacobian(z).dot(f_limit)
                LHS_limit = a.ToExpression() * det_M - J_dot_limit - l_cost(z, u_limit) * det_M
                LHS_limits.append(LHS_limit)

                lam_limit_deg = Polynomial(LHS_limit).TotalDegree() - 2
                lam_01 = prog.NewFreePolynomial(Variables(z), lam_limit_deg).ToExpression()
                lam_23 = prog.NewFreePolynomial(Variables(z), lam_limit_deg).ToExpression()
                S_ring_limit = lam_01 * (z[0]**2 + z[1]**2 - 1) + lam_23 * (z[2]**2 + z[3]**2 - 1)
                S_Jdot_limit = 0
                for i in np.arange(nz):
                    lam = prog.NewSosPolynomial(Variables(z), lam_limit_deg)[0].ToExpression()
                    S_Jdot_limit += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
                lam_u_deg = Polynomial(LHS_limit).TotalDegree() - Polynomial(u_fixed[0]).TotalDegree()
                lam_u = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                if u_limit[0] < 0:
                    Su = lam_u*(u_fixed[0] - u_limit[0])
                else:
                    Su = lam_u*(u_limit[0] - u_fixed[0])
                Sus.append(Su)
                S_ring_limits.append(S_ring_limit)
                S_Jdot_limits.append(S_Jdot_limit)
                prog.AddSosConstraint(LHS_limit + S_ring_limit + S_Jdot_limit + Su)

        options = SolverOptions()
        options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        prog.SetSolverOptions(options)
        result = Solve(prog)
        assert result.is_success()
        a_star = result.GetSolution(a).RemoveTermsWithSmallCoefficients(1e-6).ToExpression()
        J_star = Polynomial(result.GetSolution(J_expr)).RemoveTermsWithSmallCoefficients(1e-6)
        LHS_a_star = result.GetSolution(LHS)
        if actuator_saturate:
            LHS_limits_a_star = [result.GetSolution(x) for x in LHS_limits]

        prog.RemoveCost(a_cost)

        try:
            # Maximize volume beneath the value function.
            if objective=="integrate_all":
                obj = J
                for i in range(nz):
                    obj = obj.Integrate(z[i], z_min[i], z_max[i])
                prog.AddCost(-obj.ToExpression())
            elif objective=="integrate_ring":
                obj = J
                for i in non_sc_idx:
                    obj = obj.Integrate(z[i], z_min[i], z_max[i])
                c_r = 1
                cost = 0
                for m,coeff in obj.monomial_to_coefficient_map().items(): 
                    s1_deg = m.degree(z[0]) 
                    c1_deg = m.degree(z[1])
                    s2_deg = m.degree(z[2]) 
                    c2_deg = m.degree(z[3])
                    monomial_int1 = quad(lambda x: np.sin(x)**s1_deg * np.cos(x)**c1_deg, np.pi-dz, np.pi+dz)[0]
                    monomial_int2 = quad(lambda x: np.sin(x)**s2_deg * np.cos(x)**c2_deg, -dz, dz)[0]
                    cost += monomial_int1 * monomial_int2 * coeff
                poly = Polynomial(cost)
                cost_coeff = [c.Evaluate() for c in poly.monomial_to_coefficient_map().values()]
                # Make the numerics better
                cost = cost/np.max(np.abs(cost_coeff))
                prog.AddLinearCost(c_r * cost)

            # Enforce Bellman inequality.
            prog.AddSosConstraint(LHS_a_star + S_ring + S_Jdot)
            if actuator_saturate:
                for i in range(len(Sus)):
                        prog.AddSosConstraint(LHS_limits_a_star[i] + S_ring_limits[i] + S_Jdot_limits[i] + Sus[i])
            # if deg >= 0:
            #     prog.AddSosConstraint(a_star * det_M * u_denominator - J_dot - l_cost(z, u_fixed) * det_M * u_denominator + S_ring + S_Jdot)
            # else:
            #     prog.AddSosConstraint(a_star * det_M * u_denominator - J_dot - l_cost(z, u_fixed) * det_M * u_denominator + S_ring)

            result = Solve(prog)
            assert result.is_success()
            J_star = Polynomial(result.GetSolution(J_expr)).RemoveTermsWithSmallCoefficients(1e-6)
        except:
            print("Eq(11) fails...")       

        dJdz = J_star.ToExpression().Jacobian(z)
        f2_val = f2(z)
        u_star = - .5 * Rinv.dot(f2_val.T).dot(dJdz.T)

        return J_star, u_star, det_M, z_max

    z = MakeVectorVariable(nz, "z")
    K = acrobot_constrained_lqr()
    u_fixed = -K @ (z-z0)
    old_J = Polynomial(np.ones(nz)@z)
    dz = 0.01
    u_denominator = 1
    for i in range(10):
        try:
            J_star, u_fixed, u_denominator, z_max = search_upper_bound(u_fixed, dz, u_denominator=u_denominator)
            if J_star.CoefficientsAlmostEqual(old_J, 1e-3):
                print("="*10, "Converged!","="*20)
                print("Iter. ", i)
                break
            old_J = J_star

            if visualize:
                params_dict = acrobot_setup()
                params_dict["x_max"] = np.array([2* np.pi, np.pi/2, z_max[-2], z_max[-1]])
                params_dict["x_min"] = np.array([0, -np.pi/2,  -z_max[-2], -z_max[-1]])
                plot_value_function(J_star, z, params_dict, deg, file_name="saturation/iterative_{}_upper_bound_{}".format(i, objective))
            
            C = extract_polynomial_coeff_dict(J_star, z)
            data_file = open("acrobot/data/saturation/J_iterative_{}_upper_bound_deg_{}_dz_{}.pkl".format(i, deg, dz),"wb")
            pickle.dump(C, data_file)
            data_file.close()
            dz += z_incre
        except:
            dz -= z_incre
            z_incre /= 2
            if z_incre <=1e-5:
                break
            dz += z_incre
    
    print("Verified dz: ", dz)

    return J_star, z

if __name__ == '__main__':
    deg = 4
    J_star, z = acrobot_sos_iterative_upper_bound(deg, visualize=True)
