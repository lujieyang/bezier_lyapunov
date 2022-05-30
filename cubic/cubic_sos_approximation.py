import numpy as np
from cubic_polynomial_fvi import optimal_cost_to_go, plot_value_function_sos
from pydrake.solvers import mathematicalprogram as mp
from pydrake.all import (MathematicalProgram, Solve, SolverOptions, Expression,
                         CommonSolverOption, Polynomial, LinearQuadraticRegulator, Variables, Variable)
from utils import *
import pydrake.symbolic as sym

import matplotlib.pyplot as plt

def sos_lower_bound(deg):
    x0 = 0
    Q = 1
    R = 1
    # Input limits.
    U = [-1, 1]
    # State limits (region of state space where we approximate the value function).
    X = [-1, 1]
    f = lambda x, u : x - 4 * x ** 3 + u
    l = lambda x, u : Q * x ** 2 + R * u ** 2
    
    prog = MathematicalProgram()
    x = prog.NewIndeterminates(1, 'x')[0]
    u = prog.NewIndeterminates(1, 'u')[0]
    J = prog.NewFreePolynomial(Variables([x]), deg)

    # Maximize volume beneath the value function.
    J_int = J.Integrate(x, -1, 1).ToExpression()
    prog.AddLinearCost(- J_int)
    
    # S-procedure for the input limits.
    xu = Variables([x, u])
    lamx = prog.NewSosPolynomial(xu, deg)[0]
    S_procedure = lamx * Polynomial((x - X[0]) * (X[1] - x))
    
    # S-procedure for the input limits.
    lamu = prog.NewSosPolynomial(xu, deg)[0]
    S_procedure += lamu * Polynomial((u - U[0]) * (U[1] - u))
    
    # Enforce Bellman inequality.
    J_dot = J.Differentiate(x) * Polynomial(f(x, u))
    prog.AddSosConstraint(J_dot + Polynomial(l(x, u)) - S_procedure)

    # J(0) = 0.
    prog.AddLinearConstraint(J.EvaluatePartial({x: x0}).ToExpression() == 0)

    # Solve and retrieve result.
    result = Solve(prog)
    assert result.is_success()

    # retrieve value function
    J_opt = result.GetSolution(J).RemoveTermsWithSmallCoefficients(1e-6)
    cost = - result.get_optimal_cost()
    
    # plot_value_function_sos(J_opt, 0, [x], deg)
    return J_opt.ToExpression(), x

def sos_lqr_upper_bound(deg):
    x0 = 0
    Q = 1
    R = 1
    A = np.ones(1)
    B = np.ones(1)
    K, S = LinearQuadraticRegulator(A, B, np.array([Q]), np.array([R]))
    # State limits (region of state space where we approximate the value function).
    X = [-1, 1]
    f = lambda x, u : x - 4 * x ** 3 + u
    l = lambda x, u : Q * x ** 2 + R * u ** 2
    
    prog = MathematicalProgram()
    x = prog.NewIndeterminates(1, 'x')[0]
    J = prog.NewFreePolynomial(Variables([x]), deg)

    # Maximize volume beneath the value function.
    J_int = J.Integrate(x, -1, 1).ToExpression()
    prog.AddLinearCost(J_int)
    
    # S-procedure for the input limits.
    lamx = prog.NewSosPolynomial(Variables([x]), deg)[0]
    S_procedure = lamx * Polynomial((x - X[0]) * (X[1] - x))
    
    # Enforce Bellman inequality.
    u_lqr = -K[0,0] * x
    J_dot = J.Differentiate(x) * Polynomial(f(x, u_lqr))
    prog.AddSosConstraint(- J_dot - Polynomial(l(x, u_lqr)) - S_procedure)

    # J(0) = 0.
    prog.AddLinearConstraint(J.EvaluatePartial({x: x0}).ToExpression() == 0)

    # Solve and retrieve result.
    result = Solve(prog)
    assert result.is_success()

    # retrieve value function
    J_opt = result.GetSolution(J.ToExpression())
    cost = - result.get_optimal_cost()
    
    plot_value_function_sos(J_opt, 0, [x], deg)
    return J_opt, K[0,0]

def lqr_cost_to_go(K):
    x = np.linspace(-1, 1, 200)
    J = (1+K**2)*np.log(K+4*x**2-1)/8
    J -= np.min(J)
    return x, J

def sos_fixed_control_upper_bound(deg, deg_lower, approx_lower_deg=-1):
    x0 = 0
    Q = 1
    R = 1
    A = np.ones(1)
    B = np.ones(1)
    K, S = LinearQuadraticRegulator(A, B, np.array([Q]), np.array([R]))
    # State limits (region of state space where we approximate the value function).
    X = [-1, 1]
    f = lambda x, u : x - 4 * x ** 3 + u
    l = lambda x, u : Q * x ** 2 + R * u ** 2
    
    J_lower, x = sos_lower_bound(deg_lower)
    u_fixed = - J_lower.Differentiate(x)/2

    if approx_lower_deg >=0:
        prog1 = MathematicalProgram()
        prog1.AddIndeterminates(np.array([x]))
        u_approx = prog1.NewFreePolynomial(Variables([x]), approx_lower_deg)
        diff = (Polynomial(u_fixed) - u_approx)**2
        diff_int = diff.Integrate(x, -1, 1).ToExpression()
        prog1.AddQuadraticCost(diff_int, is_convex=True)
        result1 = Solve(prog1)
        assert result1.is_success()
        u_fixed = result1.GetSolution(u_approx.ToExpression())

    prog = MathematicalProgram()
    prog.AddIndeterminates(np.array([x]))
    J = prog.NewFreePolynomial(Variables([x]), deg)

    # Maximize volume beneath the value function.
    J_int = J.Integrate(x, -1, 1).ToExpression()
    prog.AddLinearCost(J_int)
    
    # S-procedure for the input limits.
    lamx = prog.NewSosPolynomial(Variables([x]), deg)[0]
    S_procedure = lamx * Polynomial((x - X[0]) * (X[1] - x))
    
    # Enforce Bellman inequality.
    J_dot = J.Differentiate(x) * Polynomial(f(x, u_fixed))
    prog.AddSosConstraint(- J_dot - Polynomial(l(x, u_fixed)) - S_procedure)

    # J(0) = 0.
    prog.AddLinearConstraint(J.EvaluatePartial({x: x0}).ToExpression() == 0)

    # Solve and retrieve result.
    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    assert result.is_success()

    # retrieve value function
    J_opt = result.GetSolution(J.ToExpression())
    cost = - result.get_optimal_cost()
    
    plot_value_function_sos(J_opt, 0, [x], deg)
    return J_opt

def sos_iterative_upper_bound(deg):
    x0 = 0
    Q = 1
    R = 1
    A = np.ones(1)
    B = np.ones(1)
    K, S = LinearQuadraticRegulator(A, B, np.array([Q]), np.array([R]))
    # State limits (region of state space where we approximate the value function).
    X = [-1, 1]
    f = lambda x, u : x - 4 * x ** 3 + u
    l = lambda x, u : Q * x ** 2 + R * u ** 2
    
    def search_upper_bound(u_fixed):
        prog = MathematicalProgram()
        prog.AddIndeterminates(np.array([x]))
        J = prog.NewFreePolynomial(Variables([x]), deg)

        # Maximize volume beneath the value function.
        J_int = J.Integrate(x, -1, 1).ToExpression()
        prog.AddLinearCost(J_int)
        
        # S-procedure for the input limits.
        lamx = prog.NewSosPolynomial(Variables([x]), deg)[0]
        S_procedure = lamx * Polynomial((x - X[0]) * (X[1] - x))
        
        # Enforce Bellman inequality.
        J_dot = J.Differentiate(x) * Polynomial(f(x, u_fixed))
        prog.AddSosConstraint(- J_dot - Polynomial(l(x, u_fixed)) - S_procedure)

        # J(0) = 0.
        prog.AddLinearConstraint(J.EvaluatePartial({x: x0}).ToExpression() == 0)

        # Solve and retrieve result.
        options = SolverOptions()
        options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        prog.SetSolverOptions(options)
        result = Solve(prog)
        assert result.is_success()

        # retrieve value function
        J_opt = result.GetSolution(J.RemoveTermsWithSmallCoefficients(1e-6))
        cost = - result.get_optimal_cost()
        u_opt = - J_opt.Differentiate(x)/2

        return J_opt, u_opt.ToExpression()

    x = Variable("x")
    u_fixed = - K[0,0] * x
    old_J = Polynomial(x)
    for i in range(50):
        J_opt, u_fixed = search_upper_bound(u_fixed)
        if J_opt.CoefficientsAlmostEqual(old_J, 1e-3):
            print("="*10, "Converged!","="*20)
            print("Iter. ", i)
            break
        if i% 5 == 0:
            plot_value_function_sos(J_opt, 0, [x], label="iter. {}".format(i))
        old_J = J_opt
    return J_opt

def sos_sample_upper_bound(deg, num_controls=5):
    x0 = 0
    Q = 1
    R = 1
    A = np.ones(1)
    B = np.ones(1)
    K, S = LinearQuadraticRegulator(A, B, np.array([Q]), np.array([R]))
    # State limits (region of state space where we approximate the value function).
    X = [-1, 1]
    U = [-1, 1]
    f = lambda x, u : x - 4 * x ** 3 + u
    l = lambda x, u : Q * x ** 2 + R * u ** 2

    def search_lambda(J):
        prog = MathematicalProgram()
        prog.AddIndeterminates(np.array([x]))
        lam_vars = np.zeros([num_controls, num_controls], dtype=Expression)
        p = np.zeros(num_controls, dtype=Expression)

        dJdx = J.Differentiate(x)
        for j in range(num_controls):
            uj = u_samples[j]
            p[j] = Polynomial(l(x, uj)) + dJdx * Polynomial(f(x, uj))
        
        sos_int = 0
        for j in range(num_controls):
            sos_expr = 0
            lamx = prog.NewSosPolynomial(Variables([x]), deg)[0]
            S_procedure = lamx * Polynomial((x - X[0]) * (X[1] - x))
            for k in range(num_controls):
                lam = prog.NewSosPolynomial(Variables([x]), deg)[0]
                lam_vars[j, k] = lam
                if j==k:
                    sos_expr += -(1+lam) * p[j]
                else:
                    sos_expr += - lam * (p[k] - p[j])
            prog.AddSosConstraint(sos_expr - S_procedure)
            sos_int += sos_expr

        sos_expr_int = sos_int.Integrate(x, -1, 1).ToExpression()
        # prog.AddLinearCost(-sos_expr_int)

        options = SolverOptions()
        options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        prog.SetSolverOptions(options)
        result = Solve(prog)
        assert result.is_success()

        return result.GetSolution(lam_vars.flatten()).reshape(num_controls, num_controls)
        

    def search_J_upper(lam):
        prog = MathematicalProgram()
        J = prog.NewFreePolynomial(Variables([x]), deg)
    
    
    x = Variable("x")
    J = Polynomial(S[0,0] * x**2)
    u_samples = np.linspace(U[0], U[1], num_controls)

    lam_vars = search_lambda(J)
    
    return J_opt

def sos_regional_upper_bound(deg):
    x0 = 0
    Q = 1
    R = 1
    A = np.ones(1)
    B = np.ones(1)
    K, S = LinearQuadraticRegulator(A, B, np.array([Q]), np.array([R]))
    # State limits (region of state space where we approximate the value function).
    X = [-1, 1]
    f = lambda x, u : x - 4 * x ** 3 + u
    l = lambda x, u : Q * x ** 2 + R * u ** 2
    
    x = Variable("x")
    u = Variable("u")
    
    # S-procedure for the input limits.
    xu = Variables([x, u])
    J = Polynomial(S[0,0] * x**2)

    def search_lambda(J):
        prog = MathematicalProgram()
        prog.AddIndeterminates(np.array(list(xu)))
        lamx = prog.NewSosPolynomial(xu, deg)[0]
        region = - Polynomial((x - X[0]) * (X[1] - x))
        S_procedure = lamx * region

        lam0 = prog.NewSosPolynomial(xu, deg)[0]
        lam1 = prog.NewSosPolynomial(xu, deg)[0]
        
        # Enforce Bellman inequality.
        J_dot = J.Differentiate(x) * Polynomial(f(x, u))
        prog.AddSosConstraint((1+lam0)*Polynomial(x**2)*region - lam1*(J_dot + Polynomial(l(x, u))))

        options = SolverOptions()
        options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        prog.SetSolverOptions(options)
        result = Solve(prog)
        assert result.is_success()

        return result.GetSolution(lam0), result.GetSolution(lam1)


    def search_J():
        prog = MathematicalProgram()
        J = prog.NewFreePolynomial(Variables([x]), deg)

        # Maximize volume beneath the value function.
        J_int = J.Integrate(x, -1, 1).ToExpression()
        prog.AddLinearCost(J_int)
            
        # J(0) = 0.
        prog.AddLinearConstraint(J.EvaluatePartial({x: x0}).ToExpression() == 0)

        # Solve and retrieve result.
        result = Solve(prog)
        assert result.is_success()

        # retrieve value function
        J_opt = result.GetSolution(J.ToExpression())
        cost = - result.get_optimal_cost()    
        plot_value_function_sos(J_opt, 0, [x], deg)
        return J_opt

    search_lambda(J)

    return J_opt

def main_fixed_control():
    x_opt, J_opt = optimal_cost_to_go()
    lower_deg = 8
    approx_lower_deg = -1
    for poly_deg in range(lower_deg, 16, 2):
        sos_fixed_control_upper_bound(poly_deg, lower_deg, approx_lower_deg)

    plt.plot(x_opt, J_opt.T, 'k', label='J*')
    plt.legend()
    plt.title("u* deg {}, approx deg {}".format(lower_deg, approx_lower_deg))
    plt.savefig("J_sos_deg_{}_approx_{}_control_upper_bound.png".format(lower_deg, approx_lower_deg)) 


if __name__ == '__main__':
    deg = 10
    sos_iterative_upper_bound(deg)

    x_opt, J_opt = optimal_cost_to_go()
    plt.plot(x_opt, J_opt.T, 'k', label='J*')
    plt.legend()
    plt.title("Iterative fixed control upper bound deg {}".format(deg))
    plt.savefig("J_sos_deg_{}_iterative_upper_bound.png".format(deg)) 

    