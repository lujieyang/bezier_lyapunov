import numpy as np
from bezier_operation import *
from pydrake.all import (MathematicalProgram, Variable, ge, CommonSolverOption,
                         Solve, SolverOptions, Variables, Polynomial, le, eq)
from pydrake.examples.pendulum import PendulumParams
import matplotlib.pyplot as plt


def pendulum_lyapunov(deg, l_deg, alpha=0, eps=1e-3):
    # [s, c, theta_dot]
    b = 0.1
    x0 = [0, 1, 0]
    f1 = np.zeros([1, 2, 2])
    f2 = np.zeros([2, 1, 2])
    f3 = np.zeros([2, 1, 2])
    f1[0, 1, 1] = 1
    f2[1, 0, 1] = -1
    f3[1, 0, 0 ] = -1
    f3[0, 0, 1] = -b

    num_V_degrees = np.array(deg)
    num_var = len(num_V_degrees)
    prog = MathematicalProgram()
    Z = tuple(np.zeros(num_var, dtype=int))
    V_var = prog.NewContinuousVariables(np.product(num_V_degrees + 1),
                                        "V")  # Drake is not working for
    # tensor, so vectorize the tensor
    V = np.array(V_var).reshape(num_V_degrees + 1)

    dVdx = bernstein_derivative(V)

    f_bern = [power_to_bernstein_poly(f) for f in [f1, f2, f3]]
    Vdot = power_to_bernstein_poly(np.zeros(np.ones(num_var, dtype=int)))
    for dim in range(num_var):
        Vdot = bernstein_add(Vdot, bernstein_mul(dVdx[dim], f_bern[dim], dtype=Variable))

    # x_square = np.zeros(np.ones(num_var, dtype=int) * 3)
    # for ind in 2*np.eye(num_var, dtype=int):
    #     x_square[tuple(ind)] = 1
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

    num_l_degrees = np.array(l_deg)
    l = prog.NewContinuousVariables(np.product(num_l_degrees + 1),
                                    "l") 
    l = np.array(l).reshape(num_l_degrees + 1)
    sc_1 = np.zeros([3, 3, 1])
    sc_1[0, 0, 0] = -1
    sc_1[2, 0, 0] = 1
    sc_1[0, 2, 0] = 1
    s_proc = bernstein_mul(l, power_to_bernstein_poly(sc_1), dtype=Variable)
    # Exponential stability with rate alpha
    # LHS = bernstein_add(bernstein_add(Vdot, alpha * V), s_proc)  # Numerical issues with entries like 3.3881317890172014e-21
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


def verify_pendulum_lyapunov(alpha=0, eps=1e-3):
    # [s, c, theta_dot]
    # V = 2.4395054818570228*1 + 0.12606196061448935*thetadot^2 + -4.8790236897694896*c + 2.4395182111018459*c^2 + -1.5077272698054742e-06*s + 
    # 0.076842107313013427*s * thetadot + 2.4654960298959656*s^2
    V = np.zeros([3, 3, 3])
    V[0, 0, 0] = 2.4395054818570228
    V[0, 0, 2] = 0.12606196061448935
    V[0, 1, 0] = -4.8790236897694896
    V[0, 2, 0] = 2.4395182111018459
    V[1, 0, 0] = -1.5077272698054742e-06
    V[1, 0, 1] = 0.076842107313013427
    V[2, 0, 0] = 2.4654960298959656
    # Mechanical Energy = (4.9050000000000002 * (1 - c) + 0.125 * pow(thetadot, 2))
    # V = np.zeros([1, 2, 3])
    # V[0, 0, 0] = 4.9050000000000002
    # V[0, 1, 0] = -4.9050000000000002
    # V[0, 0, 2] = 0.125

    V = power_to_bernstein_poly(V)
    
    b = 0.1
    x0 = [0, 1, 0]
    f1 = np.zeros([1, 2, 2])
    f2 = np.zeros([2, 1, 2])
    f3 = np.zeros([2, 1, 2])
    f1[0, 1, 1] = 1
    f2[1, 0, 1] = -1
    f3[1, 0, 0 ] = -1
    f3[0, 0, 1] = -b

    num_var = len(V.shape)

    dVdx = bernstein_derivative(V)

    f_bern = [power_to_bernstein_poly(f) for f in [f1, f2, f3]]
    Vdot = power_to_bernstein_poly(np.zeros(np.ones(num_var, dtype=int)))
    for dim in range(num_var):
        if (np.array(dVdx[dim].shape) > 0).all():
            Vdot = bernstein_add(Vdot, bernstein_mul(dVdx[dim], f_bern[dim], dtype=Variable))

    x_square = np.zeros([3, 3, 3])
    x_square[0, 0, 0] = 1
    x_square[2, 0, 0] = 1
    x_square[0, 2, 0] = 1
    x_square[0, 1, 0] = -2
    x_square[0, 0, 2] = 1
    # Reshape LHS to vector since prog.AddConstraint doesn't accept tensor
    LHS_V = bernstein_add(V, -eps*power_to_bernstein_poly(x_square)).reshape(-1, 1)

    V0 = BezierSurface(x0, V)
    # assert(V0 == 0)
    # assert((V >= 0).all())
    # assert((LHS_V >= 0).all())

    LHS = bernstein_add(Vdot, alpha * V)
    return V, Vdot


def global_pendulum():
    prog = MathematicalProgram()

    # Declare the "indeterminates", x.  These are the variables which define the
    # polynomials, but are NOT decision variables in the optimization.  We will
    # add constraints below that must hold FOR ALL x.
    s = prog.NewIndeterminates(1, "s")[0]
    c = prog.NewIndeterminates(1, "c")[0]
    thetadot = prog.NewIndeterminates(1, "thetadot")[0]
    #  x = prog.NewIndeterminates(["s", "c", "thetadot"])
    x = np.array([s, c, thetadot])

    # Write out the dynamics in terms of sin(theta), cos(theta), and thetadot
    p = PendulumParams()
    print(p.damping())
    f = [
        c * thetadot, -s * thetadot,
        (-p.damping() * thetadot - p.mass() * p.gravity() * p.length() * s) /
        (p.mass() * p.length() * p.length())
    ]

    # The fixed-point in this coordinate (because cos(0)=1).
    x0 = np.array([0, 1, 0])

    # Construct a polynomial V that contains all monomials with s,c,thetadot up
    # to degree 2.
    deg_V = 4
    V = prog.NewFreePolynomial(Variables(x), deg_V).ToExpression()

    # Add a constraint to enforce that V is strictly positive away from x0.
    # (Note that because our coordinate system is sine and cosine, V is also zero
    # at theta=2pi, etc).
    eps = 1e-4
    constraint1 = prog.AddSosConstraint(V - eps * (x - x0).dot(x - x0))

    # Construct the polynomial which is the time derivative of V.
    Vdot = V.Jacobian(x).dot(f)

    # Construct a polynomial L representing the "Lagrange multiplier".
    deg_L = 2
    L = prog.NewFreePolynomial(Variables(x), deg_L).ToExpression()

    # Add a constraint that Vdot is strictly negative away from x0 (but make an
    # exception for the upright fixed point by multipling by s^2).
    constraint2 = prog.AddSosConstraint(-Vdot - L * (s**2 + c**2 - 1) - eps *
                                        (x - x0).dot(x - x0) * s**2)
    # TODO(russt): When V is the mechanical energy, Vdot=-b*thetadot^2, so I may not need all of the multipliers here.                  

    # Add V(0) = 0 constraint
    constraint3 = prog.AddLinearConstraint(
        V.Substitute({
            s: 0,
            c: 1,
            thetadot: 0
        }) == 0)

    # Add V(theta=pi) = mgl, just to set the scale.
    constraint4 = prog.AddLinearConstraint(
        V.Substitute({
            s: 1,
            c: 0,
            thetadot: 0
        }) == p.mass() * p.gravity() * p.length())

    # Call the solver.
    result = Solve(prog)
    assert result.is_success()

    # Note that I've added mgl to the potential energy (relative to the textbook),
    # so that it would be non-negative... like the Lyapunov function.
    mgl = p.mass() * p.gravity() * p.length()
    print("Mechanical Energy = ")
    print(.5 * p.mass() * p.length()**2 * thetadot**2 + mgl * (1 - c))

    print("V =")
    Vsol = Polynomial(result.GetSolution(V))
    print(Vsol.RemoveTermsWithSmallCoefficients(1e-6))

    # Plot the results as contour plots.
    nq = 151
    nqd = 151
    q = np.linspace(-2 * np.pi, 2 * np.pi, nq)
    qd = np.linspace(-2 * mgl, 2 * mgl, nqd)
    Q, QD = np.meshgrid(q, qd)
    Energy = .5 * p.mass() * p.length()**2 * QD**2 + mgl * (1 - np.cos(Q))
    Vplot = Q.copy()
    env = {s: 0., c: 1., thetadot: 0}
    for i in range(nq):
        for j in range(nqd):
            env[s] = np.sin(Q[i, j])
            env[c] = np.cos(Q[i, j])
            env[thetadot] = QD[i, j]
            Vplot[i, j] = Vsol.Evaluate(env)

    # plt.rc("text", usetex=True)
    fig, ax = plt.subplots()
    ax.contour(Q, QD, Vplot)
    ax.contour(Q, QD, Energy, alpha=0.5, linestyles="dashed")
    ax.set_xlabel("theta")
    ax.set_ylabel("thetadot")
    ax.set_title("V (solid) and Mechanical Energy (dashed)")


def van_der_pol_lyapunov(deg, alpha=0, eps=1e-3):
    # f = [- x1, 
    #       x0 + x0^2 * x1 - x1]
    x0 = [0, 0]
    f1 = np.zeros([1, 2])
    f2 = np.zeros([3, 2])
    f1[0, 1] = -1
    f2[1, 0] = 1
    f2[2, 1] = 1
    f2[0, 1] = -1

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


def sos_cubic_lyapunov(alpha = 0.5):
    prog = MathematicalProgram()
    x = prog.NewIndeterminates(1, "x")
    f = -x + x**3

    V = prog.NewSosPolynomial(Variables(x), 2)[0].ToExpression()
    prog.AddLinearConstraint(V.Substitute({x[0]: 0}) == 0)
    prog.AddLinearConstraint(V.Substitute({x[0]: 1}) == 1)
    Vdot = V.Jacobian(x).dot(f)

    lambda_ = prog.NewSosPolynomial(Variables(x), 2)[0].ToExpression()

    prog.AddSosConstraint((-Vdot - alpha*V - lambda_ * (0.81 - x**2))[0])

    result = Solve(prog)
    assert result.is_success()

    print("Solution:", Polynomial(result.GetSolution(V)).RemoveTermsWithSmallCoefficients(1e-5).ToExpression())
    return result.GetSolution(V), result.GetSolution(Vdot), x


def cubic_lyapunov(deg, alpha=.1, x0=[0], eps=1e-3):
    # \dot x = -x + x^3/4
    f = np.array([0, -1, 0, 1/4])
    # \dot x = x - 4 * x^3
    # f = np.array([0, 1, 0, -4])

    num_V_degrees = np.array([deg])
    num_var = len(num_V_degrees)
    prog = MathematicalProgram()
    Z = tuple(np.zeros(num_var, dtype=int))
    V = prog.NewContinuousVariables(np.product(num_V_degrees + 1),
                                        "V")  # Drake is not working for
    # tensor, so vectorize the tensor
    V = np.array(V).reshape(num_V_degrees + 1)

    dVdx = bernstein_derivative(V)

    f_bern = [power_to_bernstein_poly(f)]
    Vdot = np.zeros(num_var)
    for dim in range(num_var):
        Vdot = bernstein_add(Vdot, bernstein_mul(dVdx[dim], f_bern[dim], dtype=Variable))

    x_square = np.array([0, 0, 1])
    LHS_V = bernstein_add(V, -eps*power_to_bernstein_poly(x_square))

    # prog.AddLinearConstraint(V[Z] == 0)
    V0 = BezierSurface(x0, V)
    prog.AddLinearConstraint(V0 == 0)
    prog.AddLinearConstraint(ge(LHS_V, 0))
    # Exponential stability with rate alpha
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
    dVdt = result.GetSolution(Vdot)
    return result.GetSolution(V), f_bern


def cubic_roa_lyapunov(deg, l_deg, alpha=0, eps=1e-3):
    # \dot x = x - 4 * x^3
    f = np.array([0, 1, 0, -4])

    num_V_degrees = np.array([deg])
    num_var = len(num_V_degrees)
    prog = MathematicalProgram()
    Z = tuple(np.zeros(num_var, dtype=int))
    V = prog.NewContinuousVariables(np.product(num_V_degrees + 1),
                                        "V")  # Drake is not working for
    # tensor, so vectorize the tensor
    V = np.array(V).reshape(num_V_degrees + 1)

    dVdx = bernstein_derivative(V)

    f_bern = [power_to_bernstein_poly(f)]
    Vdot = np.zeros(num_var)
    for dim in range(num_var):
        Vdot = bernstein_add(Vdot, bernstein_mul(dVdx[dim], f_bern[dim], dtype=Variable))

    x_square = np.array([0, 0, 1])
    LHS_V = bernstein_add(V, -eps*power_to_bernstein_poly(x_square))

    prog.AddLinearConstraint(V[Z] == 0)
    prog.AddLinearConstraint(ge(LHS_V, 0))
    
    # ROA -0.5<= x <= 0.5
    # roa = np.array([-0.5, 1])
    roa = np.array([-0.25, 0, 1])
    num_l_degrees = np.array(l_deg)
    l = prog.NewContinuousVariables(np.product(num_l_degrees + 1),
                                    "l") 
    l = np.array(l).reshape(num_l_degrees + 1)
    s_proc = bernstein_mul(l, power_to_bernstein_poly(roa), dtype=Variable)
    prog.AddLinearConstraint(ge(l, 0))

    # Exponential stability with rate alpha
    LHS = bernstein_add(-Vdot, s_proc)
    pos_constraint = ge(LHS, 0)
    # TODO: remove after Drake handles formula True
    for c in pos_constraint.flatten():
        if len(c.GetFreeVariables()) > 0:
            prog.AddLinearConstraint(c)

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    print(result.is_success())
    dVdt = result.GetSolution(Vdot)
    return result.GetSolution(V), f_bern


def main_cubic():
    degrees = 2
    V, f_bern = cubic_lyapunov(degrees, alpha=0.5)
    # TODO: remove after symbolic::get_constant_value gets exposed to pydrake
    dVdx = bernstein_derivative(V)
    Vdot = bernstein_mul(dVdx[0], f_bern[0])
    V *= 1e3
    Vdot *= 1e3
    plot_bezier(V, -2, 2, label='V')
    plot_bezier(Vdot, -2, 2, label='Vdot')
    plt.legend()
    plt.savefig("lyapnov.png")


def main_pendulum():
    V, f_bern = pendulum_lyapunov(2 * np.ones(3, dtype=int), 2 * np.ones(3, dtype=int),alpha=0)
    V *=1e3
    num_var = len(V.shape)
    dVdx = bernstein_derivative(V)
    Vdot = power_to_bernstein_poly(np.zeros(np.ones(num_var, dtype=int)))
    for dim in range(num_var):
        Vdot = bernstein_add(Vdot, bernstein_mul(dVdx[dim], f_bern[dim], dtype=Variable))

    bernstein_to_monomial(V)
    x2z = lambda t, td: np.array([np.sin(t), np.cos(t), td])
    plot_energy(V, -np.pi, np.pi, x2z=x2z)
    plot_energy(Vdot, -np.pi, np.pi, name="Vdot", x2z=x2z)


def main_van_der_pol():
    V, f_bern = van_der_pol_lyapunov(2 * np.ones(2, dtype=int))
    V *=1e3
    num_var = len(V.shape)
    dVdx = bernstein_derivative(V)
    Vdot = power_to_bernstein_poly(np.zeros(np.ones(num_var, dtype=int)))
    for dim in range(num_var):
        Vdot = bernstein_add(Vdot, bernstein_mul(dVdx[dim], f_bern[dim], dtype=Variable))

    bernstein_to_monomial(V)
    plot_energy(V, -2, 2)
    plot_energy(Vdot, -2, 2, name="Vdot")


def plot_sos(V, x, x_lo, x_up, label="f(x)"):
    n_breaks = 101
    x_pts = np.linspace(x_lo, x_up, n_breaks)

    y = []
    for i in range(n_breaks):
        y.append(V.Evaluate({x[0]: x_pts[i]}))

    plt.plot(x_pts, y, label=label)
    plt.title("SOS Lyapunov")


if __name__ == '__main__':
    # f = lambda x: x[0]**2 + (x[1] - 1)**2 + x[2]**2
    # check_poly_coeff_matrix(f, 3) 
    # V, Vdot, x = sos_cubic_lyapunov(0)
    # plot_sos(V, x, -1, 1, label="V")
    # plot_sos(Vdot, x, -1, 1, label="Vdot")
    # plt.savefig("sos.png")
    main_cubic()
