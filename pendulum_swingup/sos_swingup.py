import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from pydrake.examples.pendulum import (PendulumParams)
from pydrake.all import (MathematicalProgram, Variables, Solve, Polynomial, SolverOptions, CommonSolverOption)
from polynomial_integration_fvi import plot_value_function_sos

# Given the degree for the approximate value function and the polynomials
# in the S procedure, solves the SOS and returns the approximate value function
# (together with the objective of the SOS program).
def pendulum_sos_dp(deg, objective="integrate_ring", visualize=False):
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
    lam_1 = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
    S_procedure_1 = lam_1 * (z[2]**2 - 4*np.pi**2)
    lam_2 = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
    S_procedure_2 = lam_2 * (z[2]**2 - 4*np.pi**2)

    # Enforce Bellman inequality.
    J_dot = J_expr.Jacobian(z).dot(f(z, u))
    prog.AddSosConstraint(J_dot + l(z, u) + S_procedure + S_procedure_1)

    # Enforce that value function is PD
    prog.AddSosConstraint(J_expr + S_procedure_2)

    # J(z0) = 0.
    J0 = J_expr.EvaluatePartial(dict(zip(z, z0)))
    prog.AddLinearConstraint(J0 == 0)

    # Solve and retrieve result.
    prog.AddLinearConstraint(J0 == 0)
    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    result = Solve(prog)
    assert result.is_success()
    J_star = Polynomial(result.GetSolution(J_expr))

    # Solve for the optimal feedback in augmented coordinates.
    Rinv = np.linalg.inv(R)
    f2 = np.array([[0], [0], [1 / inertia]])
    dJdz = J_star.ToExpression().Jacobian(z)
    u_star = - .5 * Rinv.dot(f2.T).dot(dJdz.T)

    if visualize:
        plot_value_function_sos(J_star, u_star, z, x_min, x_max, x2z, deg, file_name="sos_{}".format(objective))
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


if __name__ == '__main__':
    J_star, u_star, z = pendulum_sos_dp(2, "integrate_ring", visualize=True)
