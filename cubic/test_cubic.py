import unittest
import numpy as np
from utils import construct_monomial_basis_from_polynomial
from pydrake.all import (MathematicalProgram, Variables, Polynomial, MultibodyPlantConfig)
from underactuated import FindResource

class TestCubic(unittest.TestCase):
    def test_calc_basis(self):
        prog = MathematicalProgram()
        z = prog.NewIndeterminates(nz, "z")
        J = prog.NewFreePolynomial(Variables(z), deg)
        J_decision_variables = np.array(list(J.decision_variables()))
        nJ = len(J_decision_variables)
        calc_basis = construct_monomial_basis_from_polynomial(J, nJ, z)

        X = np.random.rand(1, 1)
        alpha = np.random.rand(nJ)

        J_basis = calc_basis(X)
        J_basis_val = J_basis @ alpha
        J_val = J.Evaluate({**dict(zip(z, X)), **dict(zip(J_decision_variables, alpha))})

        np.testing.assert_allclose(J_basis_val, J_val)

        # Test dJdz
        J_expr = J.ToExpression()
        dJdz = J_expr.Jacobian(z)

        dJdz_poly = Polynomial(dJdz[0], z)
        calc_basis_dJdz = construct_monomial_basis_from_polynomial(dJdz_poly, nJ, z)
        dJdz_basis_val = calc_basis_dJdz(X) @ alpha
        dJdz_val = dJdz_poly.Evaluate({**dict(zip(z, X)), **dict(zip(J_decision_variables, alpha))})

        np.testing.assert_allclose(dJdz_basis_val, dJdz_val)


if __name__ == '__main__':
    nz = 1
    deg = 2
    unittest.main()