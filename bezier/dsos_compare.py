import numpy as np
from bezier_operation import *

from pydrake.all import (MathematicalProgram, Solve, Polynomial)

def p(a, b):
    # Nonnegative polynomials represented by DSOS and SDSOS can't be represented by bezier surfaces
    # with all nonnegative coefficients. Need the coefficients to be >= eps
    f = np.zeros([5, 5])
    f[4, 0] = 1
    f[0, 4] = 1
    f[3, 1] = a
    f[2, 2] = 1 - a/2 - b/2
    f[1, 3] = 2 * b

    f_bern = power_to_bernstein_poly(f)
    deg = np.array(f_bern.shape) - 1
    print(np.sum(f_bern))
    while (f_bern < 0).any():
        f_bern = bernstein_degree_elevation(f_bern, np.array([1]))
        neg_idx = np.where(f_bern <0)
        print("Degree: {}, neg coeff: {}, neg degree: {}".format(deg, f_bern[
            neg_idx], neg_idx))
        deg += 1
    print(deg-1)

def bezier_to_sos():
    # Check the other way around: if sos can represent all bezier surfaces with nonnegative coefficients.
    prog = MathematicalProgram()
    num_var = 1
    deg = 8
    x = prog.NewIndeterminates(num_var, "x")
    f = prog.NewFreePolynomial(Variables(x), deg).ToExpression()
    
    S_procedure = 0
    lam_deg = 4
    for i in range(num_var):
        lam = prog.NewSosPolynomial(Variables(np.array([x[i]])), lam_deg)[0].ToExpression()
        S_procedure += lam * ((x[i]-0.5)**2-0.5)
    prog.AddSosConstraint(f + S_procedure)

    F_deg = (deg+1) * np.ones(num_var, dtype=int)
    F = np.random.randint(0, high=10, size=F_deg)
    b = Polynomial(BezierSurface(x, F), x)

    diff = Polynomial(f,x) - b
    for monomial,coeff in diff.monomial_to_coefficient_map().items():
        print(f'monomial: {monomial}, coef: {coeff}')
        prog.AddLinearEqualityConstraint(coeff, 0)

    result = Solve(prog)
    print(result.is_success())

    if not result.is_success():
        plot_bezier(F, 0, 1)
        plt.savefig("SOS compare.png")

if __name__ == '__main__':
    bezier_to_sos()