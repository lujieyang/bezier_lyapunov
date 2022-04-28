import numpy as np
from bezier_operation import *
from pydrake.all import (MathematicalProgram, Variable, ge, CommonSolverOption,
                         Solve, SolverOptions, Variables, Polynomial, le, eq,
                         DiagramBuilder, SymbolicVectorSystem, LogVectorOutput,
                         Simulator, MosekSolver)
import pydrake.symbolic as sym
from pydrake.solvers import mathematicalprogram as mp
import dreal

def power_matching_dp(deg, extra_deg):
    print("Degree: ", deg)
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


def cubic_dp(deg, gamma=0, adv_set=[]):
    print("Degree: ", deg)
    # f(x) = x - 4 * x^3 - u
    f = np.zeros([4, 2])
    f[1, 0] = 1
    f[3, 0] = -4
    f[0, 1] = -1

    # l(x) = x^2 + u^2
    l = np.zeros([3, 3])
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

    if adv_set != []:
        for x_adv in adv_set:
            prog.AddLinearConstraint(BezierSurface([x_adv[0]], np.squeeze(J))>=0)
            prog.AddLinearConstraint(BezierSurface(x_adv, LHS)>=0)

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    print(result.is_success())
    area = -result.get_optimal_cost()

    J_opt = np.squeeze(result.GetSolution(J))
    eval_dict = dict(zip(J_var, J_opt.flatten()))
    LHS_val = np.zeros(LHS.shape)
    it = np.nditer(LHS, flags=['multi_index', 'refs_ok'])
    for _ in it:
        idx = it.multi_index
        LHS_val[idx] = LHS[idx].Evaluate(eval_dict)
    return J_opt, area, LHS_val

def smt_check(J, LHS):
    x = dreal.Variable("x")
    u = dreal.Variable("u")
    state_input_constraints = dreal.logical_and(x>=0, x<=1, u>=0, u<=1)
    J_val = BezierSurface([x], J)
    LHS_val = BezierSurface([x, u], LHS)
    config = dreal.Config()
    # config.use_polytope_in_forall = True
    # config.use_local_optimization = True
    config.precision = 1e-8
    # condition = dreal.logical_and(J_val>=0, LHS_val>=0)
    condition = dreal.logical_and(dreal.logical_imply(state_input_constraints, J_val>=0), dreal.logical_imply(state_input_constraints, LHS_val>=0))
    return dreal.CheckSatisfiability(dreal.logical_not(condition), config)

def min_optimization(LHS):
    prog = MathematicalProgram()
    xu = prog.NewContinuousVariables(2)
    prog.AddBoundingBoxConstraint(np.zeros(2)+1e-3, np.ones(2)-1e-3, xu)
    prog.AddCost(BezierSurface(xu, LHS))

    result = Solve(prog)
    print(result.is_success())
    print(result.get_solver_id().name())

    return result.GetSolution(xu), result.get_optimal_cost()


def cubic_piecewise_dp(deg, x_con=0.1):
    print("Degree: ", deg)
    # f(x) = x - 4 * x^3 - u
    f = np.zeros([4, 2])
    f[1, 0] = 1
    f[3, 0] = -4
    f[0, 1] = -1

    # l(x) = x^2 + u^2
    l = np.zeros([3, 3])
    l[2, 0] = 1
    l[0, 2] = 1

    num_J_degrees = np.array([deg, 0])
    num_var = len(num_J_degrees)
    Z = tuple(np.zeros(num_var, dtype=int))
    prog = MathematicalProgram()
    J0 = prog.NewContinuousVariables(np.product(num_J_degrees+1),
                                    "J0")  # Drake is not working for tensor, so vectorize the tensor
    J0 = np.array(J0).reshape(num_J_degrees+1)
    J1 = prog.NewContinuousVariables(np.product(num_J_degrees+1),
                                    "J1")
    J1 = np.array(J1).reshape(num_J_degrees+1)

    dJdx0 = bernstein_derivative(J0)
    dJdx1 = bernstein_derivative(J1)

    # First derivative matches at the concatenation point
    prog.AddLinearConstraint(BezierSurface([x_con], np.squeeze(dJdx0[0])) == BezierSurface([x_con], np.squeeze(dJdx1[0])))
    prog.AddLinearConstraint(BezierSurface([x_con], np.squeeze(J0)) == BezierSurface([x_con], np.squeeze(J1)))

    f_bern = power_to_bernstein_poly(f)
    l_bern = power_to_bernstein_poly(l)

    dJdx_f0 = bernstein_mul(dJdx0[0], f_bern, dtype=Variable)  # The 1st derivative correspond
    # to x
    dJdx_f1 = bernstein_mul(dJdx1[0], f_bern, dtype=Variable) 
    LHS0 = bernstein_add(l_bern, dJdx_f0)
    LHS1 = bernstein_add(l_bern, dJdx_f1)

    J_int0 = bernstein_integral(J0, 0, x_con)
    J_int1 = bernstein_integral(J1, x_con, 1)

    prog.AddLinearConstraint(J0[Z] == 0)
    pos_constraint = ge(LHS0, 0)
    for c in pos_constraint.flatten():
        if len(c.GetFreeVariables()) > 0:
            prog.AddLinearConstraint(c)
    pos_constraint = ge(LHS1, 0)
    for c in pos_constraint.flatten():
        if len(c.GetFreeVariables()) > 0:
            prog.AddLinearConstraint(c)
 
    prog.AddLinearCost(-J_int0 - J_int1)

    # options = SolverOptions()
    # options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    # prog.SetSolverOptions(options)
    result = Solve(prog)
    print(result.is_success())
    area = -result.get_optimal_cost()

    J_opt = {"J0": np.squeeze(result.GetSolution(J0)), "J1": np.squeeze(result.GetSolution(J1)), "x_con": x_con}
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

    dJdx = bernstein_derivative(J)[0]

    f1_bern = power_to_bernstein_poly(f1)
    f2_bern = power_to_bernstein_poly(f2)
    l1_bern = power_to_bernstein_poly(l1)

    dJdx_f1 = bernstein_mul(dJdx, f1_bern, dtype=Variable)  
    dJdx_f2 = bernstein_mul(dJdx, f2_bern, dtype=Variable)
    last_term = -bernstein_mul(dJdx_f2, dJdx_f2, dtype=Variable)/4
    LHS = bernstein_add(bernstein_add(l1_bern, dJdx_f1), last_term)

    prog.AddLinearConstraint(J[Z] == 0)
    prog.AddLinearConstraint(ge(J, 0))
    eq_constraint = eq(LHS, 0)
    for c in eq_constraint.flatten():
        if len(c.GetFreeVariables()) > 0:
            prog.AddConstraint(c)

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    print(result.is_success())

    J_opt = np.squeeze(result.GetSolution(J))
    return J_opt, 0


def cubic_control_affine_dp_lower_bound(deg):
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

    dJdx = bernstein_derivative(J)[0]

    f1_bern = power_to_bernstein_poly(f1)
    f2_bern = power_to_bernstein_poly(f2)
    l1_bern = power_to_bernstein_poly(l1)

    dJdx_f1 = bernstein_mul(dJdx, f1_bern, dtype=Variable)  
    dJdx_f2 = bernstein_mul(dJdx, f2_bern, dtype=Variable)
    last_term = -bernstein_mul(dJdx_f2, dJdx_f2, dtype=Variable)/4
    LHS = bernstein_add(bernstein_add(l1_bern, dJdx_f1), last_term)

    prog.AddLinearConstraint(J[Z] == 0)
    prog.AddLinearConstraint(ge(J, 0))
    eq_constraint = ge(LHS, 0)
    for c in eq_constraint.flatten():
        if len(c.GetFreeVariables()) > 0:
            prog.AddConstraint(c)

    # TODO: Change to AddQuadraticAsRotatedLorentzConeConstraint
    # Extract the quadratic form explicitly for Drake to use a convex solver
    # for i in range(len(LHS)):
    #     poly = sym.Polynomial(LHS[i])
    #     variables, map_var_to_index = sym.ExtractVariablesFromExpression(-LHS[i])
    #     Q, b, c = sym.DecomposeQuadraticPolynomial(poly, map_var_to_index)
    #     # prog.AddLorentzConeConstraint(Q, b, variables)


    J_int = bernstein_integral(J)
    prog.AddLinearCost(-J_int)

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    # solver = MosekSolver()
    result = Solve(prog)
    print(result.is_success())
    print(result.get_solver_id().name())

    J_opt = np.squeeze(result.GetSolution(J))
    return J_opt, -result.get_optimal_cost()


def cubic_control_affine_dp_lower_bound_shift(deg, eps=1e-3, x0=[0]):
    # Shift the domain of x and u to be in [0,1] and relax the positivity
    # constraint to be >= -eps.
    print("Degree: ", deg)
    f1 = np.array([-0.5, -1, -1.5, -0.5])
    f2 = np.array([-0.5])

    l1 = np.array([0, 0, 1])

    R = 1

    num_J_degrees = np.array([deg])
    num_var = len(num_J_degrees)
    Z = tuple(np.zeros(num_var, dtype=int))
    prog = MathematicalProgram()
    J_var = prog.NewContinuousVariables(np.product(num_J_degrees+1),
                                    "J")  # Drake is not working for tensor, so vectorize the tensor
    J = np.array(J_var).reshape(num_J_degrees+1)

    dJdx = bernstein_derivative(J)[0]

    f1_bern = power_to_bernstein_poly(f1)
    f2_bern = power_to_bernstein_poly(f2)
    l1_bern = power_to_bernstein_poly(l1)

    dJdx_f1 = bernstein_mul(dJdx, f1_bern, dtype=Variable)
    dJdx_f2 = bernstein_mul(dJdx, f2_bern, dtype=Variable)
    last_term = -bernstein_mul(dJdx_f2, dJdx_f2, dtype=Variable)/4
    LHS = bernstein_add(bernstein_add(l1_bern, dJdx_f1), last_term)

    # prog.AddLinearConstraint(J[Z] == 0)
    J0 = BezierSurface(x0, J)
    prog.AddLinearConstraint(J0 == 0)
    prog.AddLinearConstraint(ge(J, -eps))
    eq_constraint = ge(LHS, 0)
    for c in eq_constraint.flatten():
        if len(c.GetFreeVariables()) > 0:
            prog.AddConstraint(c)

    J_int = bernstein_integral(J)
    prog.AddLinearCost(-J_int)

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    print(result.is_success())
    print(result.get_solver_id().name())

    J_opt = np.squeeze(result.GetSolution(J))
    return J_opt, -result.get_optimal_cost()


def check_cubic_control_affine_dp(J):
    Z = tuple(np.zeros(len(J.shape), dtype=int))
    f1 = np.array([0, 1, 0, -4])
    f2 = np.array([1])

    l1 = np.array([0, 0, 1])

    dJdx = bernstein_derivative(J)[0]

    f1_bern = power_to_bernstein_poly(f1)
    f2_bern = power_to_bernstein_poly(f2)
    l1_bern = power_to_bernstein_poly(l1)

    dJdx_f1 = bernstein_mul(dJdx, f1_bern)  
    dJdx_f2 = bernstein_mul(dJdx, f2_bern)
    last_term = -bernstein_mul(dJdx_f2, dJdx_f2)/4
    LHS = bernstein_add(bernstein_add(l1_bern, dJdx_f1), last_term)

    assert(J[Z] == 0)
    assert((J >= 0).all())
    assert((LHS == 0).all())


def main_dp():
    degrees = np.arange(22, 32, 2)
    J = {deg: cubic_dp(deg) for deg in degrees}

    n_breaks = 101
    x_breaks = np.linspace(0, 1, n_breaks)

    x_opt, J_opt = optimal_cost_to_go()

    fig = plt.figure()
    for deg in degrees:
        if J[deg] is None:
            print(f'degree {deg} failed')
            continue
        label = f'Deg. {deg}'
        J_plot = [BezierSurface([xi], J[deg][0], 0, 1) for xi in x_breaks]
        plt.plot(x_breaks, J_plot, label=label)
        print(f'Degree {deg} area under curve = {J[deg][1]}')
        violation = smt_check(J[deg][0], J[deg][2])
        print(violation)
    plt.plot(x_opt, J_opt.T, 'k', label='J*')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$v$')
    plt.title('Value-function lower bound')
    # plt.title('Value-function approximation')
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


def main_piecewise_dp():
    degrees = np.arange(2, 36, 4)
    J = {deg: cubic_piecewise_dp(deg) for deg in degrees}

    x_opt, J_opt = optimal_cost_to_go()

    for deg in degrees:
        label = f'Deg. {deg}'
        plot_bezier(J[deg][0]["J0"], 0, J[deg][0]["x_con"], None)
        plot_bezier(J[deg][0]["J1"], J[deg][0]["x_con"], 1, label)
        print(f'Degree {deg} area under curve = {J[deg][1]}')
    plt.plot(x_opt, J_opt.T, 'k', label='J*')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$v$')
    plt.title('Value-function lower bound')
    plt.legend()
    plt.grid(True)

    plt.savefig('bezier_cubic_piecewise.png')


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

def iterative_dp():
    adv_set = []
    x_opt, J_opt = optimal_cost_to_go()
    plt.plot(x_opt, J_opt.T, 'k', label='J*')
    for i in range(100):
        J, area, LHS = cubic_dp(40, gamma=-1e-3, adv_set=adv_set)
        if i%10==0:
            plot_bezier(J, 0, 1, label="iter.{}".format(i))
            plt.legend()
            plt.savefig("cubic.png")
        print("area: ", area)
        xu, lhs_violation = min_optimization(LHS)
        if lhs_violation <0:
            adv_set.append(xu)
    

if __name__ == '__main__':
    # f = lambda x, u: (x+1)/2 - 4 * ((x+1)/2)**3 - (u+1)/2
    # check_1d_poly_coeff_matrix(f)
    # main_dp()
    iterative_dp()
    # J, _ = cubic_control_affine_dp(24)
    # check_cubic_control_affine_dp(J)
    # main_piecewise_dp()
