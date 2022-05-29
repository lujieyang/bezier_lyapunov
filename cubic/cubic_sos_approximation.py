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
    J_opt = result.GetSolution(J.ToExpression())
    cost = - result.get_optimal_cost()
    
    plot_value_function_sos(J_opt, 0, [x], deg)
    return J_opt, cost

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
    return J_opt, cost

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
    J = Polynomial(3 * S[0,0] * x**2)

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
        prog.AddSosConstraint((1+lam0)*Polynomial(x**2)*region - lam1*(J_dot + Polynomial(l(x, u))) + S_procedure)

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

if __name__ == '__main__':
    x_opt, J_opt = optimal_cost_to_go()
    for poly_deg in range(16, 18, 2):
        sos_regional_upper_bound(poly_deg)
    plt.plot(x_opt, J_opt.T, 'k', label='J*')
    plt.legend()
    plt.savefig("J_sos_lower_bound.png") 

    