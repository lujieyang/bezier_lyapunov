import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from pydrake.examples.pendulum import (PendulumParams)
from pydrake.all import (MathematicalProgram, Variables, Solve, Polynomial)

# Given the degree for the approximate value function and the polynomials
# in the S procedure, solves the SOS and returns the approximate value function
# (together with the objective of the SOS program).
def pendulum_sos_dp(deg):
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

    # Set up optimization.
    prog = MathematicalProgram()
    z = prog.NewIndeterminates(nz, 'z')
    u = prog.NewIndeterminates(nu, 'u')
    J = prog.NewFreePolynomial(Variables(z), deg)
    J_expr = J.ToExpression()

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

    # Solve for the optimal feedback in augmented coordinates.
    Rinv = np.linalg.inv(R)
    f2 = np.array([[0], [0], [1 / inertia]])
    dJdz = J_star.ToExpression().Jacobian(z)
    u_star = - .5 * Rinv.dot(f2.T).dot(dJdz.T)

    X1, X2 = np.meshgrid(np.linspace(x_min[0], x_max[0], 51),
                        np.linspace(x_min[1], x_max[1], 51))
    X = np.vstack((X1.flatten(), X2.flatten()))
    Z = x2z(X)
    J = np.zeros(Z.shape[1])
    U = np.zeros(Z.shape[1])
    for i in range(Z.shape[1]):
        J[i] = J_star.Evaluate({z[0]: Z[0, i], z[1]: Z[1, i], z[2]: Z[2, i]})
        U[i] = u_star[0].Evaluate({z[0]: Z[0, i], z[1]: Z[1, i], z[2]: Z[2, i]})

    fig = plt.figure(figsize=(9, 4))
    ax = fig.subplots()
    ax.set_xlabel("q")
    ax.set_ylabel("qdot")
    ax.set_title("Cost-to-Go")
    im = ax.imshow(J.reshape(X1.shape),
            cmap=cm.jet, aspect='auto',
            extent=(x_min[0], x_max[0], x_min[1], x_max[1]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.savefig("figures/fvi/pendulum_swingup.png")

    fig = plt.figure(figsize=(9, 4))
    ax = fig.subplots()
    ax.set_xlabel("q")
    ax.set_ylabel("qdot")
    ax.set_title("Policy")
    im = ax.imshow(U.reshape(X1.shape),
            cmap=cm.jet, aspect='auto',
            extent=(x_min[0], x_max[0], x_min[1], x_max[1]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.savefig("figures/fvi/pendulum_swingup_policy.png")

    return J_star, u_star, z


if __name__ == '__main__':
    J_star, u_star, z = pendulum_sos_dp(deg=14)
