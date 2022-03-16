import numpy as np
from bezier_operation import *
from pydrake.all import (MathematicalProgram, Variable, ge, CommonSolverOption,
                         Solve, SolverOptions, Variables, Polynomial, le, eq,
                         DiagramBuilder, SymbolicVectorSystem, LogVectorOutput,
                         Simulator)

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


def cubic_control_affine_dp(deg):
    print("Degree: ", deg)
    f1 = np.array([0, 1, 0, -4])
    f2 = np.array([1])

    l1 = np.array([0, 0, 1])

    R = 1

    num_J_degrees = np.array([deg])
    num_var = len(num_J_degrees)
    Z = tuple(np.zeros(num_var, dtype=int))
    prog = MathematicalProgram()
    J_var = prog.NewContinuousVariables(np.product(num_J_degrees+1),
                                    "J")  # Drake is not working for tensor, so vectorize the tensor
    J = np.array(J_var).reshape(num_J_degrees+1)

    # x_square = np.zeros([3, 1])
    # x_square[2, 0] = 1
    # J = bernstein_mul(J, power_to_bernstein_poly(x_square), dtype=Variable)

    dJdx = bernstein_derivative(J)[0]

    f1_bern = power_to_bernstein_poly(f1)
    f2_bern = power_to_bernstein_poly(f2)
    l1_bern = power_to_bernstein_poly(l1)

    dJdx_f1 = bernstein_mul(dJdx, f1_bern, dtype=Variable)  
    dJdx_f2 = bernstein_mul(dJdx, f2_bern, dtype=Variable)
    last_term = -bernstein_mul(dJdx_f2, dJdx_f2, dtype=Variable)/4
    LHS = bernstein_add(bernstein_add(l1_bern, dJdx_f1), last_term)

    J_int = bernstein_integral(J)

    prog.AddLinearConstraint(J[Z] == 0)
    # prog.AddLinearConstraint(ge(J, 0))
    eq_constraint = eq(LHS, 0)
    for c in eq_constraint.flatten():
        if len(c.GetFreeVariables()) > 0:
            prog.AddConstraint(c)
 
    # prog.AddLinearCost(-J_int)

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
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


if __name__ == '__main__':
    main_dp()