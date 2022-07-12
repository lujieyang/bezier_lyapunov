import numpy as np
from scipy.integrate import quad
from utils import load_polynomial, save_polynomial
from pydrake.examples.pendulum import (PendulumParams, PendulumPlant, PendulumInput)
from pydrake.all import (MathematicalProgram, Variables, Solve, Polynomial, SolverOptions, CommonSolverOption, Linearize, LinearQuadraticRegulator)
from polynomial_integration_fvi import plot_value_function_sos

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
    z_max = x2z(x_max)
    z_min = x2z(x_min)

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
        return nz, f, f2, Rinv, z0, l

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

    # S procedure for s^2 + c^2 = 1.
    lam = prog.NewFreePolynomial(Variables(z), deg).ToExpression()
    S_procedure = lam * (z[0]**2 + z[1]**2 - 1)
    lam_2_Jdot = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
    S_procedure_1 = lam_2_Jdot * (z[2]**2 - 4*np.pi**2)
    lam_2_J = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
    S_procedure_2 = lam_2_J * (z[2]**2 - 4*np.pi**2)

    # Enforce Bellman inequality.
    J_dot = J_expr.Jacobian(z).dot(f(z, u))
    if deg <= 4:
        prog.AddSosConstraint(J_dot + l(z, u) + S_procedure + S_procedure_1)
    else:
        prog.AddSosConstraint(J_dot + l(z, u) + S_procedure)

    # Enforce that value function is PD
    if deg <= 4:
        prog.AddSosConstraint(J_expr + S_procedure_2)
    else:
        prog.AddSosConstraint(J_expr)

    # J(z0) = 0.
    J0 = J_expr.EvaluatePartial(dict(zip(z, z0)))
    prog.AddLinearConstraint(J0 == 0)

    # Solve and retrieve result.
    # options = SolverOptions()
    # options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    # prog.SetSolverOptions(options)
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

def pendulum_sos_upper_bound(deg, deg_lower, objective="integrate_ring", visualize=False):
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
    Q = np.diag([1, 1, 1]) * 10
    R = np.diag([1])
    def l(z, u):
        return (z - z0).dot(Q).dot(z - z0) + u.dot(R).dot(u)

    # Fixed control law from lower bound
    J_lower, u_fixed, z = pendulum_sos_lower_bound(deg_lower)

    # Check if u_fixed is stabilizing
    dJdz = J_lower.ToExpression().Jacobian(z)
    xdot = f(z, u_fixed)
    Jdot = dJdz.dot(xdot)
    prog0 = MathematicalProgram()
    prog0.AddIndeterminates(z)
    lam = prog0.NewFreePolynomial(Variables(z), deg).ToExpression()
    S_procedure = lam * (z[0]**2 + z[1]**2 - 1)
    lam_1 = prog0.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
    S_procedure_1 = lam_1 * (z[2]**2 - 4*np.pi**2)
    prog0.AddSosConstraint(-Jdot + S_procedure + S_procedure_1)
    result0 = Solve(prog0)
    assert result0.is_success()

    # Set up optimization.        
    prog = MathematicalProgram()
    prog.AddIndeterminates(z)
    J = prog.NewFreePolynomial(Variables(z), deg)
    J_expr = J.ToExpression()

    a = prog.NewSosPolynomial(Variables(z), deg)[0]

    # Maximize volume beneath the value function.
    if objective=="integrate_all":
        obj = J
        for i in range(nz):
            obj = obj.Integrate(z[i], z_min[i], z_max[i])
        prog.AddCost(-obj.ToExpression())
    elif objective=="integrate_ring":
        obj = J.Integrate(z[-1], z_min[-1], z_max[-1]) + 1e3 * a.Integrate(z[-1], z_min[-1], z_max[-1])
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

    # S procedure for s^2 + c^2 = 1.
    lam = prog.NewFreePolynomial(Variables(z), deg).ToExpression()
    S_procedure = lam * (z[0]**2 + z[1]**2 - 1)
    lam_1 = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
    S_procedure_1 = lam_1 * (z[2]**2 - 4*np.pi**2)
    lam_2 = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
    S_procedure_2 = lam_2 * (z[2]**2 - 4*np.pi**2)

    # Enforce Bellman inequality.
    J_dot = J_expr.Jacobian(z).dot(f(z, u_fixed))
    prog.AddSosConstraint(a.ToExpression() - J_dot - l(z, u_fixed) + S_procedure + S_procedure_1)

    # Enforce that value function is PD
    prog.AddSosConstraint(J_expr + S_procedure_2)

    # Enforce l(x,u)-a(x) is PD
    u = prog.NewIndeterminates(nu, 'u')
    lam_3 = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
    S_procedure_3 = lam_3 * (z[2]**2 - 4*np.pi**2)
    prog.AddSosConstraint(l(z,u) - a.ToExpression() + S_procedure_3)

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
    a_star = result.GetSolution(a).RemoveTermsWithSmallCoefficients(1e-6)
    l_val = Polynomial(result.GetSolution(l(z, u_fixed)))

    # Solve for the optimal feedback in augmented coordinates.
    Rinv = np.linalg.inv(R)
    f2 = np.array([[0], [0], [1 / inertia]])
    dJdz = J_star.ToExpression().Jacobian(z)
    u_star = - .5 * Rinv.dot(f2.T).dot(dJdz.T)

    if visualize:
        plot_value_function_sos(J_star, u_star, z, x_min, x_max, x2z, deg, file_name="sos_upper_bound_{}".format(objective))
    return J_star, u_star, z

def pendulum_sos_upper_bound_relaxed(deg, deg_lower, objective="integrate_ring", visualize=False):
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
    Q = np.diag([1, 1, 1]) * 10
    R = np.diag([1])
    def l(z, u):
        return (z - z0).dot(Q).dot(z - z0) + u.dot(R).dot(u)

    # Fixed control law from lower bound
    J_lower, u_fixed, z = pendulum_sos_lower_bound(deg_lower)

    # Check if u_fixed is stabilizing
    dJdz = J_lower.ToExpression().Jacobian(z)
    xdot = f(z, u_fixed)
    Jdot = dJdz.dot(xdot)
    prog0 = MathematicalProgram()
    prog0.AddIndeterminates(z)
    lam = prog0.NewFreePolynomial(Variables(z), deg).ToExpression()
    S_procedure = lam * (z[0]**2 + z[1]**2 - 1)
    lam_1 = prog0.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
    S_procedure_1 = lam_1 * (z[2]**2 - 4*np.pi**2)
    prog0.AddSosConstraint(-Jdot + S_procedure + S_procedure_1)
    result0 = Solve(prog0)
    assert result0.is_success()

    # Set up optimization.        
    prog = MathematicalProgram()
    prog.AddIndeterminates(z)
    J = prog.NewFreePolynomial(Variables(z), deg)
    J_expr = J.ToExpression()

    a = prog.NewSosPolynomial(Variables(z), deg)[0]

    # Minimize volume beneath the a(x).
    obj = a.Integrate(z[-1], z_min[-1], z_max[-1])
    c_r = 1
    cost = 0
    for monomial,coeff in obj.monomial_to_coefficient_map().items(): 
        s_deg = monomial.degree(z[0]) 
        c_deg = monomial.degree(z[1])
        monomial_int = quad(lambda x: np.sin(x)**s_deg * np.cos(x)**c_deg, 0, 2*np.pi)[0]
        if np.abs(monomial_int) <=1e-5:
            monomial_int = 0
        cost += monomial_int * coeff
    a_cost = prog.AddLinearCost(c_r * cost)

    # S procedure for s^2 + c^2 = 1.
    lam = prog.NewFreePolynomial(Variables(z), deg).ToExpression()
    S_procedure = lam * (z[0]**2 + z[1]**2 - 1)
    lam_1 = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
    S_procedure_1 = lam_1 * (z[2]**2 - 4*np.pi**2)
    lam_2 = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
    S_procedure_2 = lam_2 * (z[2]**2 - 4*np.pi**2)

    # Enforce Bellman inequality.
    J_dot = J_expr.Jacobian(z).dot(f(z, u_fixed))
    prog.AddSosConstraint(a.ToExpression() - J_dot - l(z, u_fixed) + S_procedure + S_procedure_1)

    # Enforce that value function is PD
    prog.AddSosConstraint(J_expr + S_procedure_2)

    # Enforce l(x,u)-a(x) is PD
    u = prog.NewIndeterminates(nu, 'u')
    lam_3 = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
    S_procedure_3 = lam_3 * (z[2]**2 - 4*np.pi**2)
    prog.AddSosConstraint(l(z,u) - a.ToExpression() + S_procedure_3)

    # J(z0) = 0.
    J0 = J_expr.EvaluatePartial(dict(zip(z, z0)))
    prog.AddLinearConstraint(J0 == 0)

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
    prog.AddSosConstraint(a_star - J_dot - l(z, u_fixed) + S_procedure + S_procedure_1)

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

def pendulum_lower_bound_roa():
    nz, f, f2, Rinv, z0, l = pendulum_sos_lower_bound(2, test=True)
    prog = MathematicalProgram()
    z = prog.NewIndeterminates(nz, "z")
    V = load_polynomial(z, "pendulum_swingup/data/J_upper_deg_2.pkl")
    dVdz = V.Jacobian(z)
    u_star = - .5 * Rinv.dot(f2.T).dot(dVdz.T)
    f_val = f(z, u_star)
    V_dot = dVdz.dot(f_val)

    lhs_deg = 4
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

def verify_hjb_inequality_on_roa(rho=84):
    nz, f, f2, Rinv, z0, l = pendulum_sos_lower_bound(2, test=True)
    prog = MathematicalProgram()
    z = prog.NewIndeterminates(nz, "z")
    J = load_polynomial(z, "pendulum_swingup/data/J_upper_deg_2.pkl")
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
    # pendulum_lower_bound_roa()
    # verify_hjb_inequality_on_roa()
    # J_star, u_star, z = pendulum_sos_lower_bound(2, "integrate_ring", visualize=True)
    J_star, u_star, z = pendulum_sos_upper_bound_relaxed(2, 2, "integrate_ring", visualize=True)
