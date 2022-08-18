import numpy as np
import os
from scipy.integrate import quad
from utils import load_polynomial, save_polynomial, construct_monomial_basis_from_polynomial
from pydrake.examples.pendulum import (PendulumParams)
from pydrake.all import (MathematicalProgram, Variables, Solve, Polynomial, SolverOptions, CommonSolverOption, 
MakeVectorVariable, LinearQuadraticRegulator, Jacobian)
from polynomial_integration_fvi import plot_value_function_sos
import mcint

# Given the degree for the approximate value function and the polynomials
# in the S procedure, solves the SOS and returns the approximate value function
# (together with the objective of the SOS program).
def pendulum_sos_lower_bound(deg, objective="integrate_ring", visualize=False, test=False):
    # System dimensions. Here:
    # x = [theta, theta_dot]
    # z = [sin(theta), cos(theta), theta_dot]
    nx = 2
    nz = 3
    nu = 1

    # Map from original state to augmented state.
    # Uses sympy to be able to do symbolic integration later on.
    x2z = lambda x : np.array([np.sin(x[0]), np.cos(x[0]), x[1]])

    # System dynamics in augmented state (z).
    params = PendulumParams()
    inertia = params.mass() * params.length() ** 2
    tau_g = params.mass() * params.gravity() * params.length()
    def f(z, u):
        return [
            z[1] * z[2],
            - z[0] * z[2],
            (tau_g * z[0] + u[0] - params.damping() * z[2]) / inertia
        ]

    # State limits (region of state space where we approximate the value function).
    x_max = np.array([np.pi, 2*np.pi])
    x_min = - x_max
    z_max = np.array([1, 1, x_max[-1]])
    z_min = -z_max

    # Equilibrium point in both the system coordinates.
    x0 = np.array([0, 0])
    z0 = x2z(x0)
        
    # Quadratic running cost in augmented state.
    Q = np.diag([1, 1, 1]) * 10
    R = np.diag([1])
    Rinv = np.linalg.inv(R)
    def l(z, u):
        return (z - z0).dot(Q).dot(z - z0) + u.dot(R).dot(u)

    f2 = np.array([[0], [0], [1 / inertia]])

    if test:
        return nz, f, f2, Rinv, z0, l, z_max

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
        obj = J.Integrate(z[-1], z_min[-1], z_max[-1])
        c_r = 1
        cost = 0
        for monomial,coeff in obj.monomial_to_coefficient_map().items(): 
            s_deg = monomial.degree(z[0]) 
            c_deg = monomial.degree(z[1])
            monomial_int = quad(lambda x: np.sin(x)**s_deg * np.cos(x)**c_deg, 0, 2*np.pi)[0]
            if np.abs(monomial_int) <=1e-5:
                monomial_int = 0
            cost += monomial_int * coeff
        prog.AddLinearCost(-c_r * cost)

    J_dot = J_expr.Jacobian(z).dot(f(z, u))
    LHS = J_dot + l(z, u)

    # S procedure for s^2 + c^2 = 1.
    lam_r = prog.NewFreePolynomial(Variables(z), deg).ToExpression()
    S_r = lam_r * (z[0]**2 + z[1]**2 - 1)
    S_Jdot = 0
    for i in range(nz):
        lam = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
        S_Jdot += lam*(z[i]-z_max[i])*(z[i]-z_min[i])

    # Enforce Bellman inequality.
    prog.AddSosConstraint(LHS + S_r + S_Jdot)

    lam_r = prog.NewFreePolynomial(Variables(z), deg).ToExpression()
    S_r = lam_r * (z[0]**2 + z[1]**2 - 1)
    S_J = 0
    for i in range(nz):
        lam = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
        S_J += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
    # Enforce that value function is PD
    prog.AddSosConstraint(J_expr + S_r + S_J)

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
    dJdz = J_star.ToExpression().Jacobian(z)
    u_star = - .5 * Rinv.dot(f2.T).dot(dJdz.T)

    # save_polynomial(J_star, z, "pendulum_swingup/data/J_lower_deg_{}.pkl".format(deg))
    if visualize:
        plot_value_function_sos(J_star, u_star, z, x_min, x_max, x2z, deg, file_name="sos_{}".format(objective))
    return J_star, u_star, z

def pendulum_lqr(z0):
    # Polynomial system not stabilizable: LQR cannot see the constraint s^2 + c^2 = 1
    params = PendulumParams()
    g = params.gravity()
    m = params.mass()
    l = params.length()
    b = params.damping()

    A = np.array([[0, z0[2], z0[1]],
                  [-z0[2], 0, -z0[0]],
                  [g/l, 0, -b/(m*l**2)]])
    B = np.array([[0], [0], [1/(m*l**2)]])
    Q = np.diag((10.,10., 10.))
    R = [1]
    K = LinearQuadraticRegulator(A, B, Q, R)[0]
    return np.squeeze(K)

def pendulum_sos_upper_bound(deg, deg_lower, objective="integrate_ring", visualize=False, eps=1e-5):
    # System dimensions. Here:
    # x = [theta, theta_dot]
    # z = [sin(theta), cos(theta), theta_dot]
    nx = 2
    nz = 3
    nu = 1

    # Map from original state to augmented state.
    # Uses sympy to be able to do symbolic integration later on.
    x2z = lambda x : np.array([np.sin(x[0]), np.cos(x[0]), x[1]])

    # System dynamics in augmented state (z).
    params = PendulumParams()
    inertia = params.mass() * params.length() ** 2
    tau_g = params.mass() * params.gravity() * params.length()
    def f(z, u):
        return [
            z[1] * z[2],
            - z[0] * z[2],
            (tau_g * z[0] + u[0] - params.damping() * z[2]) / inertia
        ]

    # State limits (region of state space where we approximate the value function).
    x_max = np.array([0.8*np.pi, 2*np.pi])
    x_min = -x_max
    if x_max[0] < 0.5*np.pi:
        z_max = np.array([np.sin(x_max[0]), 1, x_max[-1]])
        z_min = np.array([np.sin(x_min[0]), np.cos(x_max[0]), x_min[-1]])
    else:
        z_max = np.array([1, 1, x_max[-1]])
        z_min = np.array([-1, np.cos(x_max[0]), x_min[-1]])
    z_min[np.abs(z_min)<=1e-6] = 0
    assert (z_min < z_max).all()

    # Equilibrium point in both the system coordinates.
    x0 = np.array([0, 0])
    z0 = x2z(x0)
        
    # Quadratic running cost in augmented state.
    Q = np.diag([1, 1, 1]) * 10
    R = np.diag([1])
    def l(z, u):
        return (z - z0).dot(Q).dot(z - z0) + u.dot(R).dot(u)

    # Fixed control law from lower bound
    J_lower, u_fixed, z = pendulum_sos_lower_bound(deg_lower)

    # Set up optimization.        
    prog = MathematicalProgram()
    prog.AddIndeterminates(z)
    J = prog.NewFreePolynomial(Variables(z), deg)
    J_expr = J.ToExpression()

    # Maximize volume beneath the value function.
    if objective=="integrate_all":
        obj = J
        for i in range(nz):
            obj = obj.Integrate(z[i], z_min[i], z_max[i])
        prog.AddCost(-obj.ToExpression())
    elif objective=="integrate_ring":
        obj = J.Integrate(z[-1], z_min[-1], z_max[-1])
        c_r = 1
        cost = 0
        for monomial,coeff in obj.monomial_to_coefficient_map().items(): 
            s_deg = monomial.degree(z[0]) 
            c_deg = monomial.degree(z[1])
            monomial_int = quad(lambda x: np.sin(x)**s_deg * np.cos(x)**c_deg, 0, 2*np.pi)[0]
            if np.abs(monomial_int) <=1e-5:
                monomial_int = 0
            cost += monomial_int * coeff
        prog.AddLinearCost(c_r * cost)

    # Enforce Bellman inequality.
    J_dot = J_expr.Jacobian(z).dot(f(z, u_fixed))
    LHS = - J_dot - l(z, u_fixed)
    # S procedure for s^2 + c^2 = 1.
    lam_ring = prog.NewFreePolynomial(Variables(z), deg+4).ToExpression()
    S_ring = lam_ring * (z[0]**2 + z[1]**2 - 1)
    S_Jdot = 0
    for i in range(nz):
        lam = prog.NewSosPolynomial(Variables(z), deg+4)[0].ToExpression()
        S_Jdot += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
    prog.AddSosConstraint(LHS + S_ring + S_Jdot)

    lam_r = prog.NewFreePolynomial(Variables(z), deg+4).ToExpression()
    S_r = lam_r * (z[0]**2 + z[1]**2 - 1)
    S_J = 0
    # for i in range(nz):
    #     lam = prog.NewSosPolynomial(Variables(z), deg+4)[0].ToExpression()
    #     S_J += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
    # Enforce that value function is PD
    prog.AddSosConstraint(J_expr - eps * z.dot(z) + S_r + S_J)

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
    f2 = np.array([[0], [0], [1 / inertia]])
    dJdz = J_star.ToExpression().Jacobian(z)
    u_star = - .5 * Rinv.dot(f2.T).dot(dJdz.T)

    save_polynomial(J_star, z, "pendulum_swingup/data/J_upper_deg_{}.pkl".format(deg))
    if visualize:
        plot_value_function_sos(J_star, u_star, z, x_min, x_max, x2z, deg, file_name="sos_upper_bound_{}".format(objective))
    return J_star, u_star, z

def calculate_eps():
    nz, f, f2, Rinv, z0, l, z_max = pendulum_sos_lower_bound(2, test=True)
    prog = MathematicalProgram()
    z = prog.NewIndeterminates(nz, "z")   
    J =  prog.NewFreePolynomial(Variables(z), 2)
    J_lower = load_polynomial(z, "pendulum_swingup/data/J_lower_deg_2.pkl")
    dJdz = J_lower.Jacobian(z)
    u_star = - .5 * Rinv.dot(f2.T).dot(dJdz.T)    

    nJ = len(np.array(list(J.decision_variables())))
    calc_basis = construct_monomial_basis_from_polynomial(J, nJ, z)
    m = np.squeeze(calc_basis(1* z.reshape(1, -1)))
    m_squared  = z.dot(z) #m.dot(m)

    f_val = f(z, u_star)
    d = m_squared.Jacobian(z).dot(f_val)
    eps = prog.NewContinuousVariables(1, 'eps')[0]

    lam_r = prog.NewFreePolynomial(Variables(z), 4).ToExpression()
    S_r = lam_r * (z[0]**2 + z[1]**2 - 1)

    prog.AddSosConstraint(l(z, u_star) + eps * d + S_r)
    prog.AddLinearCost(-eps)

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    assert result.is_success()    

    eps_star = result.GetSolution(eps)
    print("eps: ", eps_star)

    return eps_star

def pendulum_sos_iterative_upper_bound(deg, objective="integrate_ring", visualize=False):
    # System dimensions. Here:
    # x = [theta, theta_dot]
    # z = [sin(theta), cos(theta), theta_dot]
    nx = 2
    nz = 3
    nu = 1

    # Map from original state to augmented state.
    # Uses sympy to be able to do symbolic integration later on.
    x2z = lambda x : np.array([np.sin(x[0]), np.cos(x[0]), x[1]])

    # System dynamics in augmented state (z).
    params = PendulumParams()
    inertia = params.mass() * params.length() ** 2
    tau_g = params.mass() * params.gravity() * params.length()
    def f(z, u):
        return [
            z[1] * z[2],
            - z[0] * z[2],
            (tau_g * z[0] + u[0] - params.damping() * z[2]) / inertia
        ]

    # Equilibrium point in both the system coordinates.
    x0 = np.array([0, 0])
    z0 = x2z(x0)
        
    # Quadratic running cost in augmented state.
    Q = np.diag([1, 1, 1]) * 10
    R = np.diag([1])
    Rinv = np.linalg.inv(R)
    def l(z, u):
        return (z - z0).dot(Q).dot(z - z0) + u.dot(R).dot(u)

    f2 = np.array([[0], [0], [1 / inertia]])

    def search_upper_bound(u_fixed, z_max, z_min):
        # Set up optimization.        
        prog = MathematicalProgram()
        prog.AddIndeterminates(z)
        J = prog.NewFreePolynomial(Variables(z), deg)
        J_expr = J.ToExpression()

        # Maximize volume beneath the value function.
        if objective=="integrate_all":
            obj = J
            for i in range(nz):
                obj = obj.Integrate(z[i], z_min[i], z_max[i])
            prog.AddCost(-obj.ToExpression())
        elif objective=="integrate_ring":
            obj = J.Integrate(z[-1], z_min[-1], z_max[-1])
            c_r = 1
            cost = 0
            for monomial,coeff in obj.monomial_to_coefficient_map().items(): 
                s_deg = monomial.degree(z[0]) 
                c_deg = monomial.degree(z[1])
                monomial_int = quad(lambda x: np.sin(x)**s_deg * np.cos(x)**c_deg, 0, 2*np.pi)[0]
                if np.abs(monomial_int) <=1e-5:
                    monomial_int = 0
                cost += monomial_int * coeff
            prog.AddLinearCost(c_r * cost)

        # Enforce Bellman inequality.
        J_dot = J_expr.Jacobian(z).dot(f(z, u_fixed))
        LHS = - J_dot - l(z, u_fixed)
        # S procedure for s^2 + c^2 = 1.
        lam_ring = prog.NewFreePolynomial(Variables(z), deg+4).ToExpression()
        S_ring = lam_ring * (z[0]**2 + z[1]**2 - 1)
        S_Jdot = 0
        for i in range(nz):
            lam = prog.NewSosPolynomial(Variables(z), deg+4)[0].ToExpression()
            S_Jdot += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
        prog.AddSosConstraint(LHS + S_ring + S_Jdot)

        lam_r = prog.NewFreePolynomial(Variables(z), deg+4).ToExpression()
        S_r = lam_r * (z[0]**2 + z[1]**2 - 1)
        S_J = 0
        for i in range(nz):
            lam = prog.NewSosPolynomial(Variables(z), deg+4)[0].ToExpression()
            S_J += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
        # Enforce that value function is PD
        prog.AddSosConstraint(J_expr + S_r + S_J)

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
        dJdz = J_star.ToExpression().Jacobian(z)
        u_star = - .5 * Rinv.dot(f2.T).dot(dJdz.T)
        return J_star, u_star
    
    # State limits (region of state space where we approximate the value function).
    x_max = np.array([0.8*np.pi, 2*np.pi])
    x_min = -x_max

    x0_up = np.pi
    x0_lo = 0.4*np.pi

    z = MakeVectorVariable(nz, "z")
    J_upper = load_polynomial(z, "pendulum_swingup/data/J_upper_deg_2.pkl")
    dJdz = J_upper.Jacobian(z)
    u_fixed = - .5 * Rinv.dot(f2.T).dot(dJdz.T)

    for i in range(10):
        print("Iter.", i)
        if x_max[0] < 0.5*np.pi:
            z_max = np.array([np.sin(x_max[0]), 1, x_max[-1]])
            z_min = np.array([np.sin(x_min[0]), np.cos(x_max[0]), x_min[-1]])
        else:
            z_max = np.array([1, 1, x_max[-1]])
            z_min = np.array([-1, np.cos(x_max[0]), x_min[-1]])
        z_min[np.abs(z_min)<=1e-6] = 0
        assert (z_min < z_max).all()

        try:
            J_star, u_fixed = search_upper_bound(u_fixed, z_max, z_min) 
            x0_lo = x_max[0]
            save_polynomial(J_star, z, "pendulum_swingup/data/J_upper_deg_{}_iter_{}.pkl".format(deg, i))
            if visualize:
                plot_value_function_sos(J_star, u_fixed, z, x_min, x_max, x2z, deg, file_name="iter_{}_sos_upper_bound".format(i))
                print("x0 :", x_max[0]/np.pi)
        except:
            x0_up = x_max[0]
        
        x_max[0] = (x0_lo + x0_up)/2

    return J_star, u_star, z

def pendulum_sos_upper_bound_relaxed(deg, deg_lower, objective="integrate_ring", visualize=False, roa=False):
    # System dimensions. Here:
    # x = [theta, theta_dot]
    # z = [sin(theta), cos(theta), theta_dot]
    nx = 2
    nz = 3
    nu = 1

    # Map from original state to augmented state.
    # Uses sympy to be able to do symbolic integration later on.
    x2z = lambda x : np.array([np.sin(x[0]), np.cos(x[0]), x[1]])

    # System dynamics in augmented state (z).
    params = PendulumParams()
    inertia = params.mass() * params.length() ** 2
    tau_g = params.mass() * params.gravity() * params.length()
    def f(z, u):
        return [
            z[1] * z[2],
            - z[0] * z[2],
            (tau_g * z[0] + u[0] - params.damping() * z[2]) / inertia
        ]

    # State limits (region of state space where we approximate the value function).
    x_max = np.array([np.pi, 2*np.pi])
    x_min = - x_max
    z_max = np.array([1, 1, x_max[-1]])
    z_min = -z_max

    # Equilibrium point in both the system coordinates.
    x0 = np.array([0, 0])
    z0 = x2z(x0)
        
    # Quadratic running cost in augmented state.
    Q = np.diag([1, 1, 1]) * 10
    R = np.diag([1])
    def l(z, u):
        return (z - z0).dot(Q).dot(z - z0) + u.dot(R).dot(u)

    # Fixed control law from lower bound
    J_lower, u_fixed, z = pendulum_sos_lower_bound(deg_lower)
    rho_lower = 85.6874399128998

    # Set up optimization.        
    prog = MathematicalProgram()
    prog.AddIndeterminates(z)
    J = prog.NewFreePolynomial(Variables(z), deg)
    J_expr = J.ToExpression()

    a = prog.NewSosPolynomial(Variables(z), deg)[0]

    # Minimize volume beneath the a(x).
    obj = a.Integrate(z[-1], z_min[-1], z_max[-1])
    cost = 0
    if roa:
        def sampler():
            while True:
                theta = np.random.uniform(x_min[0], x_max[0])
                thetad = np.random.uniform(x_min[1], x_max[1])
                z_val = x2z([theta, thetad])
                if J_lower.Evaluate(dict(zip(z, z_val))) <= rho_lower:
                    yield (theta, thetad)

        def integrate_subelevel_set_monte_carlo(monomial_deg, n_samples=10000):
            assert len(monomial_deg) == nz
            def integrand(x):
                assert len(x) == nx
                return np.sin(x[0])**monomial_deg[0] * np.cos(x[0])**monomial_deg[1] * x[1]**monomial_deg[2]

            result, error = mcint.integrate(integrand, sampler(), measure=1, n=n_samples)
            return result 

        cost = 0
        for monomial,coeff in a.monomial_to_coefficient_map().items(): 
            monomial_deg = []
            for i in range(nz):
                monomial_deg.append(monomial.degree(z[i])) 
            monomial_int = integrate_subelevel_set_monte_carlo(monomial_deg, n_samples=1000)
            cost += monomial_int * coeff
    else:
        for monomial,coeff in obj.monomial_to_coefficient_map().items(): 
            s_deg = monomial.degree(z[0]) 
            c_deg = monomial.degree(z[1])
            monomial_int = quad(lambda x: np.sin(x)**s_deg * np.cos(x)**c_deg, 0, 2*np.pi)[0]
            if np.abs(monomial_int) <=1e-5:
                monomial_int = 0
            cost += monomial_int * coeff
    a_cost = prog.AddLinearCost(cost)

    # Enforce Bellman inequality.
    J_dot = J_expr.Jacobian(z).dot(f(z, u_fixed))
    LHS = a.ToExpression() - J_dot - l(z, u_fixed)
    # S procedure for s^2 + c^2 = 1.
    lam_ring = prog.NewFreePolynomial(Variables(z), deg+4).ToExpression()
    S_ring = lam_ring * (z[0]**2 + z[1]**2 - 1)
    if roa:
        lam = prog.NewSosPolynomial(Variables(z), deg+4)[0].ToExpression()
        S_Jdot = lam * (J_lower.ToExpression() - rho_lower)
    else:
        S_Jdot = 0
        for i in range(nz):
            lam = prog.NewSosPolynomial(Variables(z), deg+4)[0].ToExpression()
            S_Jdot += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
    prog.AddSosConstraint(LHS + S_ring + S_Jdot)

    lam_r = prog.NewFreePolynomial(Variables(z), deg+4).ToExpression()
    S_r = lam_r * (z[0]**2 + z[1]**2 - 1)
    if roa:
        lam = prog.NewSosPolynomial(Variables(z), deg+4)[0].ToExpression()
        S_J = lam * (J_lower.ToExpression() - rho_lower)        
    else:
        S_J = 0
        for i in range(nz):
            lam = prog.NewSosPolynomial(Variables(z), deg+4)[0].ToExpression()
            S_J += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
    # Enforce that value function is PD
    prog.AddSosConstraint(J_expr + S_r + S_J)

    # Enforce l(x,u)-a(x) is PD
    u = prog.NewIndeterminates(nu, 'u')
    lam_r = prog.NewFreePolynomial(Variables(z), deg+4).ToExpression()
    S_r = lam_r * (z[0]**2 + z[1]**2 - 1)
    if roa:
        lam = prog.NewSosPolynomial(Variables(z), deg+4)[0].ToExpression()
        S_la = lam * (J_lower.ToExpression() - rho_lower)  
    else:
        S_la = 0
        for i in range(nz):
            lam = prog.NewSosPolynomial(Variables(z), deg+4)[0].ToExpression()
            S_la += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
    prog.AddSosConstraint(l(z,u) - a.ToExpression() + S_r + S_la)

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
        obj = J.Integrate(z[-1], z_min[-1], z_max[-1])
        c_r = 1
        cost = 0
        for monomial,coeff in obj.monomial_to_coefficient_map().items(): 
            s_deg = monomial.degree(z[0]) 
            c_deg = monomial.degree(z[1])
            monomial_int = quad(lambda x: np.sin(x)**s_deg * np.cos(x)**c_deg, 0, 2*np.pi)[0]
            if np.abs(monomial_int) <=1e-5:
                monomial_int = 0
            cost += monomial_int * coeff
        prog.AddLinearCost(c_r * cost)

    # Enforce Bellman inequality.
    prog.AddSosConstraint(a_star - J_dot - l(z, u_fixed) + S_ring + S_Jdot)

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
    f2 = np.array([[0], [0], [1 / inertia]])
    dJdz = J_star.ToExpression().Jacobian(z)
    u_star = - .5 * Rinv.dot(f2.T).dot(dJdz.T)

    save_polynomial(J_star, z, "pendulum_swingup/data/J_upper_{}_lower_deg_{}.pkl".format(deg, deg_lower))
    if visualize:
        plot_value_function_sos(J_star, u_star, z, x_min, x_max, x2z, deg, file_name="sos_upper_bound_relaxed_{}".format(objective))
    return J_star, u_star, z

def pendulum_sos_control_affine_dp(deg):
    # System dimensions. Here:
    # x = [theta, theta_dot]
    # z = [sin(theta), cos(theta), theta_dot]
    nx = 2
    nz = 3
    nu = 1

    # Map from original state to augmented state.
    # Uses sympy to be able to do symbolic integration later on.
    x2z = lambda x : np.array([np.sin(x[0]), np.cos(x[0]), x[1]])

    # System dynamics in augmented state (z).
    params = PendulumParams()
    inertia = params.mass() * params.length() ** 2
    tau_g = params.mass() * params.gravity() * params.length()
    def f(z, u):
        return [
            z[1] * z[2],
            - z[0] * z[2],
            (tau_g * z[0] + u[0] - params.damping() * z[2]) / inertia
        ]

    # State limits (region of state space where we approximate the value function).
    x_max = np.array([np.pi, 2*np.pi])
    x_min = - x_max
    z_max = x2z(x_max)
    z_min = x2z(x_min)

    # Equilibrium point in both the system coordinates.
    x0 = np.array([0, 0])
    z0 = x2z(x0)
        
    # Quadratic running cost in augmented state.
    Q = np.diag([1, 1, 1]) * 50
    R = np.diag([1])
    def l(z, u):
        return (z - z0).dot(Q).dot(z - z0) + u.dot(R).dot(u)

    for i in range(3):
        print("Iter: ", i)
        # Set up optimization.
        prog = MathematicalProgram()
        z = prog.NewIndeterminates(nz, 'z')
        J = prog.NewFreePolynomial(Variables(z), deg)
        J_expr = J.ToExpression()

        # Solve for the optimal feedback in augmented coordinates.
        Rinv = np.linalg.inv(R)
        f2 = np.array([[0], [0], [1 / inertia]])
        if i == 0:
            old_J = Polynomial(np.ones(nz).dot(z))
        dJdz = old_J.ToExpression().Jacobian(z)
        u = - .5 * Rinv.dot(f2.T).dot(dJdz.T)

        # Maximize volume beneath the value function.
        obj = J
        for i in range(nz):
            obj = obj.Integrate(z[i], z_min[i], z_max[i])
        prog.AddLinearCost(- obj.ToExpression())

        # S procedure for s^2 + c^2 = 1.
        lam = prog.NewFreePolynomial(Variables(z), deg).ToExpression()
        S_procedure = lam * (z[0]**2 + z[1]**2 - 1)
        lam_1 = prog.NewFreePolynomial(Variables(z), deg).ToExpression()
        S_procedure_1 = lam_1 * (z[0]**2 + z[1]**2 - 1)

        # Enforce Bellman inequality.
        J_dot = J_expr.Jacobian(z).dot(f(z, u))
        prog.AddSosConstraint(J_dot + l(z, u) + S_procedure)

        # Enforce that value function is PD
        prog.AddSosConstraint(J_expr + S_procedure_1)

        # J(z0) = 0.
        J0 = J_expr.EvaluatePartial(dict(zip(z, z0)))
        prog.AddLinearConstraint(J0 == 0)

        # Solve and retrieve result.
        result = Solve(prog)
        assert result.is_success()
        J_star = Polynomial(result.GetSolution(J_expr))

        if J_star.CoefficientsAlmostEqual(old_J, 1e-5):
            break
        else:
            diff = Polynomial(J_star.ToExpression(), z)-Polynomial(old_J.ToExpression(), z)
            coeff_diff = []
            for monomial,coeff in diff.monomial_to_coefficient_map().items():
                coeff_diff.append(coeff)
                # print(f'monomial: {monomial}, coef: {coeff}')
            # print("coefficient difference: ", coeff_diff)

        old_J = J_star

    dJdz = J_star.ToExpression().Jacobian(z)
    u_star = - .5 * Rinv.dot(f2.T).dot(dJdz.T)
    plot_value_function_sos(J_star, u_star, z, x_min, x_max, x2z, deg, directory="sos")
    return J_star, u_star, z

def maximize_roa(V_deg=2):
    nz, f, f2, Rinv, z0, l, z_max = pendulum_sos_lower_bound(2, test=True)
    z = MakeVectorVariable(nz, "z")

    J = load_polynomial(z, "pendulum_swingup/data/J_upper_deg_2.pkl")
    dJdz = J.Jacobian(z)
    u_star = - .5 * Rinv.dot(f2.T).dot(dJdz.T)
    f_cl = f(z, u_star)
    rho0 = pendulum_lower_bound_roa()

    V_scale = 0
    for j in range(nz):
        ej = np.zeros(nz)
        ej[j] = 1
        V_scale += J.Evaluate(dict(zip(z, ej)))

    def search_lambda(V_star, rho_star):
        prog = MathematicalProgram()
        prog.AddIndeterminates(z)
        dVdz = V_star.Jacobian(z)
        V_dot = dVdz.dot(f_cl)

        lhs_deg = Polynomial(V_dot).TotalDegree()
        lam_rho_deg = lhs_deg
        lam_rho_Vdot = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_rho_deg/2)*2))[0].ToExpression()
        S_rho = lam_rho_Vdot * (V_star - rho_star)
        lam_r_Vdot = prog.NewFreePolynomial(Variables(z), lhs_deg).ToExpression()
        S_r = lam_r_Vdot*(z[0]**2+z[1]**2-1)
        S_Vdot = 0
        # for i in range(nz):
        #     lam = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_rho_deg/2)*2))[0].ToExpression()
        #     S_Vdot += lam * (z[i]+z_max[i]) * (z[i]-z_max[i])
        prog.AddSosConstraint(-V_dot + S_r + S_rho + S_Vdot)

        lam_rho_deg = Polynomial(V_star).TotalDegree()
        lam_rho_V = prog.NewSosPolynomial(Variables(z), lam_rho_deg)[0].ToExpression()
        S_rho = lam_rho_V * (V_star - rho_star)
        lam_r_V = prog.NewFreePolynomial(Variables(z), lam_rho_deg).ToExpression()
        S_r = lam_r_V*(z[0]**2+z[1]**2-1)
        S_V = 0
        # for i in range(nz):
        #     lam = prog.NewSosPolynomial(Variables(z), lam_rho_deg)[0].ToExpression()
        #     S_V += lam * (z[i]+z_max[i]) * (z[i]-z_max[i])
        prog.AddSosConstraint(V_star + S_r + S_rho + S_V)   

        # options = SolverOptions()
        # options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        # prog.SetSolverOptions(options)
        result = Solve(prog)
        assert result.is_success()

        lam_rho_star = []
        for lam in [lam_rho_Vdot, lam_rho_V, lam_r_Vdot, lam_r_V]:
            lam = result.GetSolution(lam)
            lam_rho_star.append(Polynomial(lam).RemoveTermsWithSmallCoefficients(1e-6).ToExpression())
        return lam_rho_star

    def search_V_rho(lam_rho_star):
        prog = MathematicalProgram()
        prog.AddIndeterminates(z)
        V = prog.NewFreePolynomial(Variables(z), V_deg).ToExpression()
        rho = prog.NewContinuousVariables(1, 'rho')[0]

        dVdz = V.Jacobian(z)
        V_dot = dVdz.dot(f_cl)
        S_rho = lam_rho_star[0] * (V - rho)
        S_r = lam_rho_star[2]*(z[0]**2+z[1]**2-1)
        # lhs_deg = Polynomial(V_dot).TotalDegree()
        # lam_r = prog.NewFreePolynomial(Variables(z), lhs_deg).ToExpression()
        # S_r = lam_r*(z[0]**2+z[1]**2-1)
        S_Vdot = 0
        # for i in range(nz):
        #     lam = prog.NewSosPolynomial(Variables(z),  int(np.ceil(lhs_deg/2)*2))[0].ToExpression()
        #     S_Vdot += lam * (z[i]+z_max[i]) * (z[i]-z_max[i])
        prog.AddSosConstraint(-V_dot + S_r + S_rho + S_Vdot)

        S_rho = lam_rho_star[1] * (V - rho)
        S_r = lam_rho_star[3]*(z[0]**2+z[1]**2-1)
        # lam_r = prog.NewFreePolynomial(Variables(z), V_deg).ToExpression()
        # S_r = lam_r*(z[0]**2+z[1]**2-1)
        S_V = 0
        # for i in range(nz):
        #     lam = prog.NewSosPolynomial(Variables(z), V_deg)[0].ToExpression()
        #     S_V += lam * (z[i]+z_max[i]) * (z[i]-z_max[i])
        prog.AddSosConstraint(V + S_r + S_rho + S_V)   

        # Fix V scale
        V_ej = 0
        for j in range(nz):
            ej = np.zeros(nz)
            ej[j] = 1
            V_ej += V.EvaluatePartial(dict(zip(z, ej)))        
        prog.AddLinearConstraint(V_ej == 1)
        
        V0 = V.EvaluatePartial(dict(zip(z, z0)))
        prog.AddLinearConstraint(V0 == 0)
        prog.AddLinearCost(-rho)

        # options = SolverOptions()
        # options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        # prog.SetSolverOptions(options)
        result = Solve(prog)
        assert result.is_success()

        return result.GetSolution(V), result.GetSolution(rho)
    
    V_star = J/V_scale
    rho_star = rho0/V_scale
    try:
        for i in range(500):
            if i % 50 == 0:
                print("Iter. ", i)
            lam_rho_star = search_lambda(V_star, rho_star)
            V_star, rho_star = search_V_rho(lam_rho_star)
    except:
        pass
    
    save_polynomial(Polynomial(V_star), z, "pendulum_swingup/data/V_roa_deg_{}.pkl".format(V_deg))
    print("rho: ", rho_star)

    return V_star, rho_star

def pendulum_lower_bound_roa():
    nz, f, f2, Rinv, z0, l, z_max = pendulum_sos_lower_bound(2, test=True)
    prog = MathematicalProgram()
    z = prog.NewIndeterminates(nz, "z")
    x_max = [0.95*np.pi, 2*np.pi]
    V = load_polynomial(z, "pendulum_swingup/data/{}/J_upper_deg_2.pkl".format(x_max))
    dVdz = V.Jacobian(z)
    u_star = - .5 * Rinv.dot(f2.T).dot(dVdz.T)
    f_val = f(z, u_star)
    V_dot = dVdz.dot(f_val)

    lhs_deg = 2 + 6
    lam_deg = lhs_deg - Polynomial(V_dot).TotalDegree()
    lam = prog.NewFreePolynomial(Variables(z), lam_deg).ToExpression()
    lam_r_deg = lhs_deg - 2
    lam_r_01 = prog.NewFreePolynomial(Variables(z), lam_r_deg).ToExpression()

    rho = prog.NewContinuousVariables(1, 'rho')[0]

    z_squared = pow((z-z0).dot(z-z0), 1)
    prog.AddSosConstraint(z_squared*(V - rho) - lam*V_dot + lam_r_01 * (z[0]**2 + z[1]**2 - 1))
    prog.AddLinearCost(-rho)
    # prog.AddLinearConstraint(rho<=33)

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    assert result.is_success()

    rho_star = result.GetSolution(rho)
    print("rho: ", rho_star)
    return rho_star

def verify_hjb_inequality_on_roa(rho=70):
    nz, f, f2, Rinv, z0, l, z_max = pendulum_sos_lower_bound(2, test=True)
    prog = MathematicalProgram()
    z = prog.NewIndeterminates(nz, "z")
    J = load_polynomial(z, "pendulum_swingup/data/J_upper_6_lower_deg_2.pkl")
    dJdz = J.Jacobian(z)
    u_star = - .5 * Rinv.dot(f2.T).dot(dJdz.T)
    f_val = f(z, u_star)
    J_dot = dJdz.dot(f_val)

    LHS = -J_dot
    # LHS = -(J_dot + l(z, u_star))
    lhs_deg = Polynomial(LHS).TotalDegree() + 4
    lam_deg = lhs_deg - 2
    lam_r_01 = prog.NewFreePolynomial(Variables(z), lam_deg).ToExpression()
    lam_rho = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_deg/2)*2))[0].ToExpression()
    prog.AddSosConstraint(LHS + lam_r_01 * (z[0]**2 + z[1]**2 - 1) + lam_rho*(J-rho))

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    assert result.is_success()

if __name__ == '__main__':
    # eps = calculate_eps()
    # pendulum_lower_bound_roa()
    # verify_hjb_inequality_on_roa()
    # J_star, u_star, z = pendulum_sos_lower_bound(2, "integrate_ring", visualize=True)
    J_star, u_star, z = pendulum_sos_upper_bound(2, 2, "integrate_ring", visualize=True)