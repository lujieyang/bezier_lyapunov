from turtle import shape
import cvxpy as cp
import numpy as np
from pydrake.all import (MathematicalProgram, Variable, ge, CommonSolverOption,
                         Solve, SolverOptions, Variables, Polynomial, le,
                         DiagramBuilder, SymbolicVectorSystem, LogVectorOutput,
                         Simulator)

from scipy.special import comb
import itertools
import matplotlib.pyplot as plt


def power_to_bernstein_poly(X):
    N = np.array(X.shape) - 1
    num_var = len(N)
    Z = np.zeros(num_var, dtype=int)  # multi-index with all 0's

    P = np.zeros(N + 1)
    S = construct_S(Z, N)
    for I in S:
        Js = construct_S(Z, I)
        PI = 0
        for J in Js:
            PI += bernstein_comb(I, J)/bernstein_comb(N, J) * X[J]
        P[I] = PI
    return P


def construct_S(L, U):
    # Construct the set S with all the combinations which are
    # greater than or equal to the multi-index L and smaller than 
    # or equal to U
    iterables = [range(t, k+1) for t, k in zip(L, U)]
    return itertools.product(*iterables)


def bernstein_comb(I, J):
    p = 1
    for k in range(len(I)):
        p *= comb(I[k], J[k])
    return p


def bernstein_add(F, G):
    Nf = np.array(F.shape) - 1
    Ng = np.array(G.shape) - 1
    if (Nf == Ng).all():
        H = F + G
    else:
        NE = np.maximum(Nf, Ng)
        F_ele = bernstein_degree_elevation(F, NE - Nf)
        G_ele = bernstein_degree_elevation(G, NE - Ng)
        H = F_ele + G_ele
    return H


def bernstein_degree_elevation(F, E):
    N = np.array(F.shape) - 1
    num_var = len(N)
    Z = np.zeros(num_var, dtype=int) # multi-index with all 0's
    dtype = type(F[tuple(Z)])
    H = np.zeros(N + E + 1, dtype=dtype)
    S = construct_S(Z, N + E)
    for K in S:
        D = np.maximum(Z, K - E)
        U = np.minimum(N, K)
        Ls = construct_S(D, U)
        HK = 0
        for L in Ls:
            K_L = tuple(np.array(K) - np.array(L))
            N_E = tuple(np.array(N) + np.array(E))
            HK += bernstein_comb(N, L) * bernstein_comb(E, K_L) / bernstein_comb(N_E, K) * F[L]
        H[K] = HK
    return H


def bernstein_mul(F, G, dtype=float):
    Nf = np.array(F.shape) - 1
    Ng = np.array(G.shape) - 1
    num_var = len(F.shape)
    N = Nf + Ng
    H = np.zeros(N+1, dtype=dtype)
    Z = np.zeros(num_var, dtype=int)  # multi-index with all 0's
    S = construct_S(Z, N)

    for K in S:
        D = np.maximum(Z, K - Ng)
        U = np.minimum(Nf, K)
        Ls = construct_S(D, U)
        HK = 0
        for L in Ls:
            K_L = tuple(np.array(K) - np.array(L))
            HK += bernstein_comb(Nf, L) * bernstein_comb(Ng, K_L) / bernstein_comb(N, K) * F[L] * G[K_L]
        H[K] = HK
    return H


def bernstein_derivative(X):
    N = np.array(X.shape) - 1
    num_var = len(N)
    Z = np.zeros(num_var, dtype=int)
    dtype = type(X[tuple(Z)])
    D = []

    for i in range(num_var):
        Ni = np.copy(N)
        Ni[i] -= 1
        Di = np.zeros(Ni + 1, dtype=dtype)
        S = construct_S(Z, N)
        for I in S:
            I_1 = np.copy(I)
            I_1[i] -= 1
            di = N[i]   # degree of i-th variable
            if (np.array(I) <= Ni).all():
                Di[I] += - di * X[I]
            if (I_1>=0).all():               
                I_1 = tuple(I_1)  # change np array into tuple for indexing               
                Di[I_1] += di * X[I]
        D.append(Di)
    return D


def bernstein_integral(X):
    dim = np.array(X.shape)
    return np.sum(X)/np.product(dim)


def BezierCurve(t, x):
    if len(x) == 1:
        return x[0]
    return (1-t)*BezierCurve(t, x[:-1]) + t*BezierCurve(t, x[1:])


def BernsteinPolynomial(t, i, n):
    c = np.math.factorial(n) / (np.math.factorial(i) * np.math.factorial(n - i))
    return c * t**i * (1-t)**(n-i)


# a multi-dimensional Bezier surface in the variables x with degrees K.shape-1
# (and coefficients K).
def BezierSurface(x, K):
    assert len(x) == len(K.shape)
    it = np.nditer(K, flags=['multi_index', 'refs_ok'])
    p = 0
    for k in it:
        b = np.copy(k)
        for dim, idx in enumerate(it.multi_index):
            b *= BernsteinPolynomial(x[dim], idx, K.shape[dim]-1)
        p += b
    return p

def pendulum_lyapunov(deg, l_deg, alpha=0, eps=1e-6):
    b = 10
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

    x_square = np.zeros(np.ones(num_var, dtype=int) * 3)
    for ind in 2*np.eye(num_var, dtype=int):
        x_square[tuple(ind)] = 1
    # Reshape LHS to vector since prog.AddConstraint doesn't accept tensor
    LHS_V = bernstein_add(V, -eps*power_to_bernstein_poly(x_square)).reshape(-1, 1)

    prog.AddLinearConstraint(V[Z] == 0)
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
    LHS = bernstein_add(bernstein_add(Vdot, alpha * V), s_proc)

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

def cubic_lyapunov(deg, alpha=.1, eps=1e-6):
    # \dot x = -x + x^3
    f = np.array([0, -1, 0, 1])
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

    prog.AddLinearConstraint(V[Z] == 0)
    prog.AddLinearConstraint(ge(LHS_V, 0))
    # Exponential stability with rate alpha
    LHS = bernstein_add(Vdot, alpha * V)
    neg_constraint = le(LHS, 0)
    # TODO: remove after Drake handles formula True
    for c in neg_constraint.flatten():
        if len(c.GetFreeVariables()) > 0:
            prog.AddLinearConstraint(c)

    # V_int = bernstein_integral(V)

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    print(result.is_success())
    dVdt = result.GetSolution(Vdot)
    return result.GetSolution(V), f_bern


def main_lyapunov():
    degrees = 4
    V, f_bern = cubic_lyapunov(degrees, alpha=0)
    # TODO: remove after symbolic::get_constant_value gets exposed to pydrake
    dVdx = bernstein_derivative(V)
    Vdot = bernstein_mul(dVdx[0], f_bern[0])
    V *= 1e7
    Vdot *= 1e7
    plot_bezier(V, -.5, .5, label='V')
    plot_bezier(Vdot, -.5, .5, label='Vdot')
    plt.legend()
    plt.savefig("lyapnov.png")


def power_matching_dp(deg, extra_deg):
    print("Degree: ", deg)
    # Scalar dynamics.
    # f = lambda x, u: (x+1)/2 - 4 * ((x+1)/2) ** 3 - (u+1)/2
    # # Quadratic running cost.
    # l = lambda x, u: ((x+1)/2) ** 2 + ((u+1)/2) ** 2

    # Scalar dynamics.
    f = lambda x, u: x - 4 * x ** 3 + 2 * u -1
    # Quadratic running cost.
    l = lambda x, u: x ** 2 + u ** 2

    # Set up SOS program.
    prog = MathematicalProgram()
    x = Variable('x')
    u = Variable('u')
    a = prog.NewContinuousVariables(deg+1, 'a')
    J = Polynomial(BezierCurve(x, a), Variables([x]))
    prog.AddLinearConstraint(J.EvaluatePartial({x: 0}).ToExpression() == 0)

    # Maximize volume beneath the value function.
    J_int = J.Integrate(x, 0, 1).ToExpression()
    prog.AddLinearCost(-J_int)

    xu = Variables([x, u])
    J_dot = J.Differentiate(x) * Polynomial(f(x, u), xu)

    p = J_dot + Polynomial(l(x, u), xu)
    num_coeffs = 1
    coeffs_shape = []
    for v in xu:
        d = p.Degree(v) + extra_deg
        num_coeffs *= d + 1
        coeffs_shape.append(d + 1)
    k = prog.NewContinuousVariables(num_coeffs, 'k')
    prog.AddBoundingBoxConstraint(0, 1000, k)
    b = Polynomial(BezierSurface([x, u], k.reshape(coeffs_shape)), xu)
    diff = p - b
    for monomial, coeff in diff.monomial_to_coefficient_map().items():
        prog.AddLinearEqualityConstraint(coeff, 0)

    # Solve and retrieve result.
    # options = SolverOptions()
    # options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    # prog.SetSolverOptions(options)
    result = Solve(prog)
    print(result.get_solver_id().name())
    if not result.is_success():
        return None, None

    # retrieve value function
    J_opt_expr = result.GetSolution(J.ToExpression())
    print(f'J_opt = {J_opt_expr}')
    J_opt = lambda x_eval: J_opt_expr.Evaluate({x: x_eval})
    area = -result.get_optimal_cost()
    J_coeff = result.GetSolution(a)

    return J_opt, area


def check_poly_coeff_matrix():
    f = lambda x, u: (x + 1) / 2 - 4 * ((x + 1) / 2) ** 3 - (u + 1) / 2
    l = lambda x, u: ((x + 1) / 2) ** 2 + ((u + 1) / 2) ** 2
    # f = lambda x, u: x - 4 * x ** 3 - u
    # l = lambda x, u: x ** 2 + u ** 2

    x = Variable('x')
    u = Variable('u')
    xu = Variables([x, u])

    print(Polynomial(f(x, u), xu).monomial_to_coefficient_map())
    print(Polynomial(l(x, u), xu).monomial_to_coefficient_map())


def cubic_dp(deg, gamma=0):
    print("Degree: ", deg)
    # Scale x and u to be in [0, 1]
    f = np.zeros([4, 2])
    # f[0, 0] = -.5
    # f[1, 0] = -1
    # f[2, 0] = -1.5
    # f[3, 0] = -.5
    # f[0, 1] = -1/2
    f[1, 0] = 1
    f[3, 0] = -4
    f[0, 1] = -1

    l = np.zeros([3, 3])
    # l[0, 0] = .5
    # l[1, 0] = .5
    # l[2, 0] = 1/4
    # l[0, 1] = .5
    # l[0, 2] = 1/4
    l[2, 0] = 1
    l[0, 2] = 1

    num_J_degrees = np.array([deg, 0])
    num_var = len(num_J_degrees)
    Z = tuple(np.zeros(num_var, dtype=int))
    prog = MathematicalProgram()
    J_var = prog.NewContinuousVariables(np.product(num_J_degrees+1),
                                    "J")  # Drake is not working for tensor, so vectorize the tensor
    J = np.array(J_var).reshape(num_J_degrees+1)

    # x_square = np.zeros([3, 1])
    # x_square[2, 0] = 1
    # J = bernstein_mul(J, power_to_bernstein_poly(x_square), dtype=Variable)

    dJdx = bernstein_derivative(J)

    f_bern = power_to_bernstein_poly(f)
    l_bern = power_to_bernstein_poly(l)

    dJdx_f = bernstein_mul(dJdx[0], f_bern, dtype=Variable)  # The 1st derivative correspond
    # to x
    LHS = bernstein_add(l_bern, dJdx_f)

    J_int = bernstein_integral(J)

    prog.AddLinearConstraint(J[Z] == 0)
    pos_constraint = ge(LHS, gamma)
    for c in pos_constraint.flatten():
        if len(c.GetFreeVariables()) > 0:
            prog.AddLinearConstraint(c)
 
    prog.AddLinearCost(-J_int)

    # options = SolverOptions()
    # options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    # prog.SetSolverOptions(options)
    result = Solve(prog)
    print(result.is_success())
    area = -result.get_optimal_cost()

    J_opt = np.squeeze(result.GetSolution(J))
    return J_opt, area


def main_dp():
    degrees = np.arange(250, 1001, 250)
    J = {deg: cubic_dp(deg) for deg in degrees}

    n_breaks = 101
    x_breaks = np.linspace(0, 1, n_breaks)

    x_opt, J_opt = optimal_cost_to_go()

    fig = plt.figure()
    plt.plot(x_opt, J_opt.T, 'k', label='J*')
    for deg in degrees:
        if J[deg] is None:
            print(f'degree {deg} failed')
            continue
        label = f'Deg. {deg}'
        J_plot = [BezierSurface([xi], J[deg][0]) for xi in x_breaks]
        plt.plot(x_breaks, J_plot, label=label)
        print(f'Degree {deg} area under curve = {J[deg][1]}')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$v$')
    plt.title('Value-function lower bound')
    plt.legend()
    plt.grid(True)

    plt.savefig('bezier_cubic.png')

    # J_pm = {deg: power_matching_dp(deg, 0) for deg in degrees}
    # fig = plt.figure()
    # plt.plot(x_opt, J_opt.T, 'k', label='J*')
    # for deg in degrees:
    #     label = f'Deg. {deg}'
    #     J_plot = [J_pm[deg][0](xi) for xi in x_breaks]
    #     plt.plot(x_breaks, J_plot, label=label)
    #     print(f'Degree {deg} area under curve = {J[deg][1]}')
    # plt.xlabel(r'$x$')
    # plt.ylabel(r'$v$')
    # plt.title('Value-function lower bound')
    # plt.legend()
    # plt.grid(True)
    #
    # plt.savefig('bezier_cubic_power_matching.png')


def optimal_cost_to_go():
    # Numerically integrate the solution to the HJB to get the "true" optimal
    # cost-to-go; this only works in the scalar case when we can compute the
    # optimal policy explicitly as a function of dJdx.
    x = Variable('x')
    J = Variable('J')
    builder = DiagramBuilder()
    sys = builder.AddSystem(
        SymbolicVectorSystem(time=x,
                             state=[J],
                             dynamics=[
                                 2 * (x - 4 * x**3)
                                 + 2 * x * np.sqrt(2 - 8 * x**2 + 16 * x**4)
                             ], output=[J]))
    logger = LogVectorOutput(sys.get_output_port(), builder)
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    # Set J(0) = 0
    context.SetTime(0.0)
    context.SetContinuousState([0.0])
    simulator.AdvanceTo(1.0)
    log = logger.FindLog(context)
    return log.sample_times(), log.data()


def check_coeff_positivity():
    # f(x) = (x-0.2)^2
    f = np.array([0.04, -0.4, 1])

    f_bern = power_to_bernstein_poly(f)
    deg = len(f_bern) - 1
    while (f_bern < 0).any():
        f_bern = bernstein_degree_elevation(f_bern, np.array([1]))
        deg += 1
        if deg % 10 == 0:
            print("Degree: {}, min coeff: {}".format(deg, np.min(f_bern)))
            # plot_bezier(f_bern, -1, 1)
    print(deg)


def verify_half_domain():
    # f(x) = x^2
    f = np.array([0, 0, 1])

    f_bern = power_to_bernstein_poly(f)
    plot_bezier(f_bern, -1, 1)


def plot_bezier(f_bern, x_lo, x_up, label="f(x)"):
    deg = len(f_bern) - 1

    n_breaks = 101
    x = np.linspace(x_lo, x_up, n_breaks)
    y = np.zeros(n_breaks)

    for d in range(deg + 1):
        y += f_bern[d] * BernsteinPolynomial(x, d, deg)

    plt.plot(x, y, label=label)
    # plt.show()


def plot_energy(V):
    n_points = 51
    theta = np.linspace(-np.pi, np.pi, n_points)
    thetadot = np.linspace(-10, 10, n_points)

    E = np.zeros([n_points, n_points])
    for i in range(n_points):
        t = theta[i]
        for j in range(n_points):
            td = thetadot[j]
            E[i, j] = BezierSurface(np.array([np.sin(t), np.cos(t), td]), V)
    [X, Y] = np.meshgrid(theta, thetadot)
    plt.contourf(X, Y, E)
    plt.colorbar()
    plt.savefig("Energy.png")



if __name__ == '__main__':
    # main_dp()
    main_lyapunov()
    # V, f_bern = pendulum_lyapunov(4 * np.ones(3, dtype=int), [2, 2, 2],alpha=0)
    # V *=1e6
    # plot_energy(V)











