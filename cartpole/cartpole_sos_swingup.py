import numpy as np
from scipy.integrate import quad
from utils import extract_polynomial_coeff_dict, reconstruct_polynomial_from_dict
import pickle
from pydrake.all import (MathematicalProgram, Variables, Expression, Solve, Polynomial, SolverOptions, CommonSolverOption, 
Linearize, LinearQuadraticRegulator, DiagramBuilder, AddMultibodyPlantSceneGraph, Parser, MakeVectorVariable)

# from underactuated import FindResource
from cartpole_swingup_fvi import plot_value_function, cartpole_setup

# Given the degree for the approximate value function and the polynomials
# in the S procedure, solves the SOS and returns the approximate value function
# (together with the objective of the SOS program).
def cartpole_sos_lower_bound(deg, objective="integrate_ring", visualize=False, test=False):
    nz = 5
    nq = 2
    nx = 2 * nq
    nu = 1

    mc = 10
    mp = 1
    l = .5
    g = 9.81
    
    # Map from original state to augmented state.
    # Uses sympy to be able to do symbolic integration later on.
    # x = (x, theta, xdot, thetadot)
    # z = (x, s, c, xdot, thetadot)
    x2z = lambda x : np.array([x[0], np.sin(x[1]), np.cos(x[1]), x[2], x[3]])

    def T(z, dtype=Expression):
        assert len(z) == nz
        T = np.zeros([nz, nx], dtype=dtype)
        T[0, 0] = 1
        T[1, 1] = z[2]
        T[2, 1] = -z[1]
        T[3, 2] = 1
        T[4, 3] = 1
        return T

    def f(z, u, T):
        assert len(z) == nz
        s = z[1]
        c = z[2]
        qdot = z[-nq:]
        denominator = (mc+mp*s**2)
        f_val = np.zeros(nx, dtype=Expression)
        f_val[:nq] = qdot * denominator
        f_val[2] = (u + mp*s*(l*qdot[1]**2+g*c))[0]
        f_val[3] = ((-u*c - mp*l*qdot[1]**2*c*s - (mc+mp)*g*s)/l)[0]
        return T @ f_val, denominator
    
    def f2(z, T, dtype=Expression):
        assert len(z) == nz
        s = z[1]
        c = z[2]
        f2_val = np.zeros([nx, nu], dtype=dtype)
        f2_val[2, :] = 1/(mc+mp*s**2)
        f2_val[3, :] =-c/(mc+mp*s**2)/l
        return T@f2_val
    
    # State limits (region of state space where we approximate the value function).
    z_max = np.array([2, 1, 1, 5, 5])
    z_min = -z_max

    # Equilibrium point in both the system coordinates.
    x0 = np.array([0, np.pi, 0, 0])
    z0 = x2z(x0)
    z0[np.abs(z0)<=1e-6] = 0
        
    # Quadratic running cost in augmented state.
    # Q = np.diag([100, 1000, 1000, 500, 500])
    Q = np.diag([200, 2e3, 2e3, 1e3, 1e3])
    R = np.diag([1]) 
    def l_cost(z, u):
        return (z - z0).dot(Q).dot(z - z0) + u.dot(R).dot(u)

    Rinv = np.linalg.inv(R)

    if test:
        return nz, f, f2, T, z0, Rinv

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
        obj = J.Integrate(z[0], z_min[0], z_max[0])
        for i in range(3, nz):
            obj = obj.Integrate(z[i], z_min[i], z_max[i])
        c_r = 1
        cost = 0
        for monomial,coeff in obj.monomial_to_coefficient_map().items(): 
            s1_deg = monomial.degree(z[1]) 
            c1_deg = monomial.degree(z[2])
            monomial_int1 = quad(lambda x: np.sin(x)**s1_deg * np.cos(x)**c1_deg, 0, 2*np.pi)[0]
            if np.abs(monomial_int1) <=1e-5:
                monomial_int1 = 0
            cost += monomial_int1 * coeff
        poly = Polynomial(cost)
        cost_coeff = [c.Evaluate() for c in poly.monomial_to_coefficient_map().values()]
        # Make the numerics better
        prog.AddLinearCost(-c_r * cost/np.max(np.abs(cost_coeff)))
        # prog.AddLinearCost(-c_r * cost)
    # prog.AddQuadraticCost(np.sum(np.array(list(J.decision_variables()))**2), is_convex=True)

    # S procedure for s^2 + c^2 = 1.
    lam = prog.NewFreePolynomial(Variables(z), deg).ToExpression()
    S_procedure = lam * (z[1]**2 + z[2]**2 - 1)
    # S procedure for compact domain 
    lam_Jdot_0 = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
    S_Jdot_0 = lam_Jdot_0 * (z[0]-z_max[0]) * (z[0]-z_min[0])
    lam_Jdot_3 = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
    S_Jdot_3 = lam_Jdot_3 * (z[3]-z_max[3]) * (z[3]-z_min[3])
    lam_Jdot_4 = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
    S_Jdot_4 = lam_Jdot_4 * (z[4]-z_max[4]) * (z[4]-z_min[4])
    
    lam_J_0 = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
    S_J_0 = lam_J_0 * (z[0]-z_max[0]) * (z[0]-z_min[0])
    lam_J_3 = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
    S_J_3 = lam_J_3 * (z[3]-z_max[3]) * (z[3]-z_min[3])
    lam_J_4 = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
    S_J_4 = lam_J_4 * (z[4]-z_max[4]) * (z[4]-z_min[4])


    # Enforce Bellman inequality.
    T_val = T(z)
    f_val, denominator = f(z, u, T_val)
    J_dot = J_expr.Jacobian(z).dot(f_val)
    if deg <= 100:
        prog.AddSosConstraint(J_dot + l_cost(z, u) * denominator + S_procedure + S_Jdot_0 + S_Jdot_3 + S_Jdot_4)
    else:
        prog.AddSosConstraint(J_dot + l_cost(z, u) * denominator + S_procedure)

    # Enforce that value function is PD
    if deg <= 100:
        prog.AddSosConstraint(J_expr + S_J_0 + S_J_3 + S_J_4)
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

    # Solve for the optimal feedback in augmented coordinates.
    Rinv = np.linalg.inv(R)
    T_val = T(z)
    f2_val = f2(z, T_val)
    dJdz = J_star.ToExpression().Jacobian(z)
    u_star = - .5 * Rinv.dot(f2_val.T).dot(dJdz.T)

    if visualize:
        params_dict = cartpole_setup()
        params_dict["x_max"] = np.array([2, 2*np.pi, 5, 5])
        params_dict["x_min"] = np.array([-2, 0 *np.pi, -5, -5])
        plot_value_function(J_star, z, params_dict, deg, file_name="sos/lower_bound_{}".format(objective))
    return J_star, u_star, z

def cartpole_lqr(nz=5):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    file_name = FindResource("models/cartpole.urdf")
    Parser(plant).AddModelFromFile(file_name)
    plant.Finalize()

    x0 = np.array([0, np.pi, 0, 0])
    context = plant.CreateDefaultContext()
    context.get_mutable_continuous_state_vector()\
                .SetFromVector(x0)
    plant.get_actuation_input_port().FixValue(context, np.array([0]))


    linearized_plant = Linearize(
        plant,
        context,
        input_port_index=plant.get_actuation_input_port().get_index(), output_port_index=plant.get_state_output_port().get_index())

    A = linearized_plant.A()
    B = linearized_plant.B()
    Q = np.diag((2., 2., 1., 1.))
    R = [1]
    K, S = LinearQuadraticRegulator(A, B, Q, R)
    K = np.squeeze(K)
    Kz = np.zeros(nz)
    Kz[:2] = K[:2]
    Kz[-2:] = K[-2:] 
    S = np.insert(np.insert(S, 2, np.zeros(4), axis=0), 2, np.zeros(5), axis=1)
    return Kz, S

def cartpole_constrained_lqr():
    nz = 5
    nu = 1
    mc = 10
    mp = 1
    l = .5
    g = 9.81

    A = np.zeros([nz, nz])
    B = np.zeros([nz, nu])
    A[0, 3] = 1
    A[1, -1] = -1
    A[3, 1] = -mp*g/mc
    A[4, 1] = -g*(mc+mp)/(l*mc)
    B[3, :] = 1/mc
    B[4, :] = 1/(l*mc)
    F = np.array([0, 0, -1, 0, 0])
    Q = np.diag([10, 10, 10, 1, 1])
    R = np.diag([1]) 
    K, S = LinearQuadraticRegulator(A, B, Q, R, F=F.reshape(1, nz))
    return np.squeeze(K), S

def cartpole_lqr_ROA():
    nz, f, f2, T, z0, Rinv = cartpole_sos_iterative_upper_bound(2, 2, test=True)
    K, S = cartpole_constrained_lqr()
    prog = MathematicalProgram()
    z = prog.NewIndeterminates(nz, "nz")
    V = (z-z0).dot(S).dot(z-z0) + 1e-4 * (z-z0).dot(z-z0)
    u_star = np.array([-K @ (z-z0)])
    dVdz = V.Jacobian(z)
    T_val = T(z)
    # f2_val, u_denominator = f2(z, T_val)
    # u_star = - .5 * Rinv.dot(f2_val.T).dot(dVdz.T)
    u_denominator = 1
    f_val, denominator = f(z, u_star, T_val, u_denominator=u_denominator)
    V_dot = dVdz.dot(f_val)
    lam_deg = 4 + Polynomial(denominator * u_denominator).TotalDegree() - Polynomial(V_dot).TotalDegree()
    lam_deg = int(np.ceil(lam_deg/2)*2)
    # lam = prog.NewSosPolynomial(Variables(z), lam_deg)[0].ToExpression()
    lam = prog.NewFreePolynomial(Variables(z), lam_deg).ToExpression()
    rho = prog.NewContinuousVariables(1, 'rho')[0]

    prog.AddSosConstraint((z-z0).dot(z-z0)*(V - rho)*denominator*u_denominator - lam*V_dot)

    prog.AddLinearCost(-rho)

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)

    print("rho: ", result.GetSolution(rho))
    return rho

def cartpole_sos_upper_bound_relaxed(deg, deg_lower, objective="integrate_ring", visualize=False):
    nz = 5
    nq = 2
    nx = 2 * nq
    nu = 1

    mc = 10
    mp = 1
    l = .5
    g = 9.81
    
    # Map from original state to augmented state.
    # Uses sympy to be able to do symbolic integration later on.
    # x = (x, theta, xdot, thetadot)
    # z = (x, s, c, xdot, thetadot)
    x2z = lambda x : np.array([x[0], np.sin(x[1]), np.cos(x[1]), x[2], x[3]])

    def T(z, dtype=Expression):
        assert len(z) == nz
        T = np.zeros([nz, nx], dtype=dtype)
        T[0, 0] = 1
        T[1, 1] = z[2]
        T[2, 1] = -z[1]
        T[3, 2] = 1
        T[4, 3] = 1
        return T

    def f(z, u, T, dtype=Expression, u_denominator=1):
        assert len(z) == nz
        s = z[1]
        c = z[2]
        qdot = z[-nq:]
        denominator = (mc+mp*s**2)
        f_val = np.zeros(nx, dtype=dtype)
        f_val[:nq] = qdot * denominator * u_denominator
        f_val[2] = (u + mp*s*(l*qdot[1]**2+g*c) * u_denominator)[0]
        f_val[3] = ((-u*c - mp*l*qdot[1]**2*c*s * u_denominator - (mc+mp)*g*s * u_denominator)/l)[0]
        return T @ f_val, denominator
    
    def f2(z, T, dtype=Expression):
        assert len(z) == nz
        s = z[1]
        c = z[2]
        denominator = (mc+mp*s**2)
        f2_val = np.zeros([nx, nu], dtype=dtype)
        f2_val[2, :] = 1
        f2_val[3, :] =-c/l
        return T@f2_val, denominator

    # State limits (region of state space where we approximate the value function).
    z_max = np.array([2, 1, 1, 5, 5])
    z_min = -z_max

    # Equilibrium point in both the system coordinates.
    x0 = np.array([0, np.pi, 0, 0])
    z0 = x2z(x0)
    z0[np.abs(z0)<=1e-6] = 0
        
    # Quadratic running cost in augmented state.
    Q = np.diag([10, 10, 10, 1, 1])
    R = np.diag([1]) 
    def l_cost(z, u):
        return (z - z0).dot(Q).dot(z - z0) + u.dot(R).dot(u)

    Rinv = np.linalg.inv(R)

    non_sc_idx = [0, 3, 4]

    # Fixed control law from lower bound
    # Set up optimization.        
    prog = MathematicalProgram()
    z = prog.NewIndeterminates(nz, 'z')
    if deg_lower == 2:
        K, _ = cartpole_constrained_lqr()
        u_fixed = np.array([-K@(z-z0)])
        u_denominator = 1
    else:
        with open("cartpole/data/sos/J_upper_bound_lower_deg_2_deg_{}.pkl".format(deg_lower), "rb") as input_file:
            C = pickle.load(input_file)
        J_lower = reconstruct_polynomial_from_dict(C, z)
        T_val = T(z)
        f2_val, u_denominator = f2(z, T_val)
        dJdz = J_lower.Jacobian(z)
        u_fixed = - .5 * Rinv.dot(f2_val.T).dot(dJdz.T)

    J = prog.NewFreePolynomial(Variables(z), deg)
    J_expr = J.ToExpression()

    a = prog.NewSosPolynomial(Variables(z), deg)[0]

    # Minimize volume beneath the a(x).
    obj = a
    for i in non_sc_idx:
        obj = obj.Integrate(z[i], z_min[i], z_max[i])
    cost = 0
    for monomial,coeff in obj.monomial_to_coefficient_map().items(): 
        s_deg = monomial.degree(z[1]) 
        c_deg = monomial.degree(z[2])
        monomial_int = quad(lambda x: np.sin(x)**s_deg * np.cos(x)**c_deg, 0, 2*np.pi)[0]
        cost += monomial_int * coeff
    poly = Polynomial(cost)
    cost_coeff = [c.Evaluate() for c in poly.monomial_to_coefficient_map().values()]
    # Make the numerics better
    cost = Polynomial(cost/np.max(np.abs(cost_coeff))).RemoveTermsWithSmallCoefficients(1e-6).ToExpression()
    a_cost = prog.AddLinearCost(cost)

    # Enforce Bellman inequality.
    T_val = T(z)
    f_val, denominator = f(z, u_fixed, T_val, u_denominator=u_denominator)
    J_dot = J_expr.Jacobian(z).dot(f_val)
    LHS = a.ToExpression()* denominator*u_denominator  - J_dot - l_cost(z, u_fixed) * denominator*u_denominator
    
    lam_deg = Polynomial(LHS).TotalDegree() - 2
    # S procedure for s^2 + c^2 = 1.
    lam = prog.NewFreePolynomial(Variables(z), lam_deg).ToExpression()
    S_procedure = lam * (z[1]**2 + z[2]**2 - 1)

    S_Jdot = 0
    # Also constrain theta to be in [-pi/2, pi/2]
    for i in np.arange(nz):
        lam = prog.NewSosPolynomial(Variables(z), lam_deg)[0].ToExpression()
        S_Jdot += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
    prog.AddSosConstraint(LHS + S_procedure + S_Jdot)

    S_J = 0
    # Also constrain theta to be in [-pi/2, pi/2]
    for i in np.arange(nz):
        lam = prog.NewSosPolynomial(Variables(z), lam_deg)[0].ToExpression()
        S_J += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
    # Enforce that value function is PD
    prog.AddSosConstraint(J_expr + S_J)

    # Enforce l(x,u)-a(x) is PD
    u = prog.NewIndeterminates(nu, 'u')
    S_la = 0
    # Also constrain theta to be in [-pi/2, pi/2]
    for i in np.arange(nz):
        lam = prog.NewSosPolynomial(Variables(z), deg-2)[0].ToExpression()
        S_la += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
    prog.AddSosConstraint(l_cost(z,u) - a.ToExpression() + S_la)

    # J(z0) = 0.
    J0 = J_expr.EvaluatePartial(dict(zip(z, z0)))
    prog.AddLinearConstraint(J0 == 0)

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    assert result.is_success()
    a_star = result.GetSolution(a).RemoveTermsWithSmallCoefficients(1e-6).ToExpression()
    LHS_a_star = result.GetSolution(LHS)

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
        cost = 0
        for monomial,coeff in obj.monomial_to_coefficient_map().items(): 
            s_deg = monomial.degree(z[1]) 
            c_deg = monomial.degree(z[2])
            monomial_int = quad(lambda x: np.sin(x)**s_deg * np.cos(x)**c_deg, 0, 2*np.pi)[0]
            cost += monomial_int * coeff
        poly = Polynomial(cost)
        cost_coeff = [c.Evaluate() for c in poly.monomial_to_coefficient_map().values()]
        # Make the numerics better
        cost = cost/np.max(np.abs(cost_coeff))
        prog.AddLinearCost(cost)

    # Enforce Bellman inequality.
    prog.AddSosConstraint(LHS_a_star + S_procedure + S_Jdot)

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

    # Solve for the optimal feedback in augmented coordinates.
    T_val = T(z)
    f2_val, _ = f2(z, T_val)
    dJdz = J_star.ToExpression().Jacobian(z)
    u_star = - .5 * Rinv.dot(f2_val.T).dot(dJdz.T)

    if visualize:
        params_dict = cartpole_setup()
        params_dict["x_max"] = np.array([2, 2*np.pi, 5, 5])
        params_dict["x_min"] = np.array([-2, 0, -5, -5])
        plot_value_function(J_star, z, params_dict, deg, file_name="sos/upper_bound_{}_lower_deg_{}".format(objective, deg_lower))
    return J_star, z

def cartpole_sos_iterative_upper_bound(deg, deg_lower, objective="integrate_ring", visualize=False, test=False):
    nz = 5
    nq = 2
    nx = 2 * nq
    nu = 1

    mc = 10
    mp = 1
    l = .5
    g = 9.81
    
    # Map from original state to augmented state.
    # Uses sympy to be able to do symbolic integration later on.
    # x = (x, theta, xdot, thetadot)
    # z = (x, s, c, xdot, thetadot)
    x2z = lambda x : np.array([x[0], np.sin(x[1]), np.cos(x[1]), x[2], x[3]])

    def T(z, dtype=Expression):
        assert len(z) == nz
        T = np.zeros([nz, nx], dtype=dtype)
        T[0, 0] = 1
        T[1, 1] = z[2]
        T[2, 1] = -z[1]
        T[3, 2] = 1
        T[4, 3] = 1
        return T

    def f(z, u, T, dtype=Expression, u_denominator=1):
        assert len(z) == nz
        s = z[1]
        c = z[2]
        qdot = z[-nq:]
        denominator = (mc+mp*s**2)
        f_val = np.zeros(nx, dtype=dtype)
        f_val[:nq] = qdot * denominator * u_denominator
        f_val[2] = (u + mp*s*(l*qdot[1]**2+g*c) * u_denominator)[0]
        f_val[3] = ((-u*c - mp*l*qdot[1]**2*c*s * u_denominator - (mc+mp)*g*s * u_denominator)/l)[0]
        return T @ f_val, denominator
    
    def f2(z, T, dtype=Expression):
        assert len(z) == nz
        s = z[1]
        c = z[2]
        denominator = (mc+mp*s**2)
        f2_val = np.zeros([nx, nu], dtype=dtype)
        f2_val[2, :] = 1
        f2_val[3, :] =-c/l
        return T@f2_val, denominator

    # State limits (region of state space where we approximate the value function).
    z_max = np.array([2, 1, 1, 5, 5])
    z_min = -z_max

    # Equilibrium point in both the system coordinates.
    x0 = np.array([0, np.pi, 0, 0])
    z0 = x2z(x0)
    z0[np.abs(z0)<=1e-6] = 0
        
    # Quadratic running cost in augmented state.
    Q = np.diag([10, 10, 10, 1, 1])
    R = np.diag([1]) 
    def l_cost(z, u):
        return (z - z0).dot(Q).dot(z - z0) + u.dot(R).dot(u)

    Rinv = np.linalg.inv(R)

    if test:
        return nz, f, f2, T, z0, Rinv

    non_sc_idx = [0, 3, 4]

    # Fixed control law from lower bound
    # Set up optimization.        
    def search_upper_bound(u_fixed, u_denominator=1):
        prog = MathematicalProgram()
        prog.AddIndeterminates(z)
        J = prog.NewFreePolynomial(Variables(z), deg)
        J_expr = J.ToExpression()

        a = prog.NewSosPolynomial(Variables(z), deg)[0]

        # Minimize volume beneath the a(x).
        obj = a
        for i in non_sc_idx:
            obj = obj.Integrate(z[i], z_min[i], z_max[i])
        cost = 0
        for monomial,coeff in obj.monomial_to_coefficient_map().items(): 
            s_deg = monomial.degree(z[1]) 
            c_deg = monomial.degree(z[2])
            monomial_int = quad(lambda x: np.sin(x)**s_deg * np.cos(x)**c_deg, 0, 2*np.pi)[0]
            if np.abs(monomial_int) <=1e-6:
                monomial_int = 0
            cost += monomial_int * coeff
        poly = Polynomial(cost)
        cost_coeff = [c.Evaluate() for c in poly.monomial_to_coefficient_map().values()]
        # Make the numerics better
        cost = cost/np.max(np.abs(cost_coeff))
        a_cost = prog.AddLinearCost(cost)

        # Enforce Bellman inequality.
        T_val = T(z)
        f_val, denominator = f(z, u_fixed, T_val, u_denominator=u_denominator)
        J_dot = J_expr.Jacobian(z).dot(f_val)
        LHS = a.ToExpression()* denominator*u_denominator  - J_dot - l_cost(z, u_fixed) * denominator*u_denominator
        
        lam_deg = Polynomial(LHS).TotalDegree() - 2
        # S procedure for s^2 + c^2 = 1.
        lam = prog.NewFreePolynomial(Variables(z), lam_deg).ToExpression()
        S_procedure = lam * (z[1]**2 + z[2]**2 - 1)

        S_Jdot = 0
        # Also constrain theta to be in [-pi/2, pi/2]
        for i in np.arange(nz):
            lam = prog.NewSosPolynomial(Variables(z), lam_deg)[0].ToExpression()
            S_Jdot += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
        prog.AddSosConstraint(LHS + S_procedure + S_Jdot)

        S_J = 0
        # Also constrain theta to be in [-pi/2, pi/2]
        for i in np.arange(nz):
            lam = prog.NewSosPolynomial(Variables(z), lam_deg)[0].ToExpression()
            S_J += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
        # Enforce that value function is PD
        prog.AddSosConstraint(J_expr + S_J)

        # Enforce l(x,u)-a(x) is PD
        u = prog.NewIndeterminates(nu, 'u')
        S_la = 0
        # Also constrain theta to be in [-pi/2, pi/2]
        for i in np.arange(nz):
            lam = prog.NewSosPolynomial(Variables(z), deg-2)[0].ToExpression()
            S_la += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
        prog.AddSosConstraint(l_cost(z,u) - a.ToExpression() + S_la)

        # J(z0) = 0.
        J0 = J_expr.EvaluatePartial(dict(zip(z, z0)))
        prog.AddLinearConstraint(J0 == 0)

        options = SolverOptions()
        options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        prog.SetSolverOptions(options)
        result = Solve(prog)
        assert result.is_success()
        a_star = result.GetSolution(a).RemoveTermsWithSmallCoefficients(1e-6).ToExpression()
        LHS_a_star = result.GetSolution(LHS)

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
            cost = 0
            for monomial,coeff in obj.monomial_to_coefficient_map().items(): 
                s_deg = monomial.degree(z[1]) 
                c_deg = monomial.degree(z[2])
                monomial_int = quad(lambda x: np.sin(x)**s_deg * np.cos(x)**c_deg, 0, 2*np.pi)[0]
                cost += monomial_int * coeff
            poly = Polynomial(cost)
            cost_coeff = [c.Evaluate() for c in poly.monomial_to_coefficient_map().values()]
            # Make the numerics better
            cost = cost/np.max(np.abs(cost_coeff))
            prog.AddLinearCost(cost)

        # Enforce Bellman inequality.
        prog.AddSosConstraint(LHS_a_star + S_procedure + S_Jdot)

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

        # Solve for the optimal feedback in augmented coordinates.
        T_val = T(z)
        f2_val, u_denominator = f2(z, T_val)
        dJdz = J_star.ToExpression().Jacobian(z)
        u_star = - .5 * Rinv.dot(f2_val.T).dot(dJdz.T)
        return J_star, u_star, u_denominator
    
    z = MakeVectorVariable(nz, "z")
    old_J = Polynomial(np.ones(nz)@z)
    if deg_lower == 2:
        # Constrained LQR controller
        K, _ = cartpole_constrained_lqr()
        u_fixed = np.array([-K@(z-z0)])
        u_denominator = 1
    else:
        with open("cartpole/data/sos/J_lower_bound_deg_{}.pkl".format(deg_lower), "rb") as input_file:
            C = pickle.load(input_file)
        J_lower = reconstruct_polynomial_from_dict(C, z)
        T_val = T(z)
        f2_val, u_denominator = f2(z, T_val)
        dJdz = J_lower.Jacobian(z)
        u_fixed = - .5 * Rinv.dot(f2_val.T).dot(dJdz.T)

    for i in range(10):
        print("Iter.", i)
        J_star, u_fixed, u_denominator = search_upper_bound(u_fixed, u_denominator=u_denominator)
        if J_star.CoefficientsAlmostEqual(old_J, 1e-3):
            print("="*10, "Converged!","="*20)
            print("Iter. ", i)
            break
        old_J = J_star

        if visualize:
            params_dict = cartpole_setup()
            params_dict["x_max"] = np.array([2, 2*np.pi, 5, 5])
            params_dict["x_min"] = np.array([-2, 0, -5, -5])
            plot_value_function(J_star, z, params_dict, deg, file_name="sos/iterative_{}_upper_bound_{}_lower_deg_{}".format(i, objective, deg_lower))
        
        C = extract_polynomial_coeff_dict(J_star, z)
        data_file = open("cartpole/data/sos/J_iterative_{}_upper_bound_lower_deg_{}_deg_{}.pkl".format(i, deg_lower, deg),"wb")
        pickle.dump(C, data_file)
        data_file.close()

    return J_star, z

if __name__ == '__main__':
    deg = 2
    deg_lower = 2
    cartpole_lqr_ROA()
    # J_star, z = cartpole_sos_upper_bound_relaxed(deg, deg_lower, visualize=True)
    # # J_star, z = cartpole_sos_iterative_upper_bound(deg, deg_lower, visualize=True)

    # C = extract_polynomial_coeff_dict(J_star, z)
    # f = open("cartpole/data/sos/J_upper_bound_lower_deg_{}_deg_{}.pkl".format(deg_lower, deg),"wb")
    # pickle.dump(C, f)
    # f.close()
