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

        X = np.random.rand(1, nz)
        alpha = np.random.rand(nJ)

        J_basis = calc_basis(X)
        J_basis_val = J_basis @ alpha
        J_val = J.Evaluate({**dict(zip(z, np.squeeze(X))), **dict(zip(J_decision_variables, alpha))})

        np.testing.assert_allclose(J_basis_val, J_val)

        # Test dJdz
        J_expr = J.ToExpression()
        dJdz = J_expr.Jacobian(z)

        dJdz_poly = Polynomial(dJdz[0], z)
        calc_basis_dJdz = construct_monomial_basis_from_polynomial(dJdz_poly, nJ, z)
        dphi_dx = calc_basis_dJdz(X)
        dPhi_dx = np.expand_dims(dphi_dx, axis=1)
        dict_val = {**dict(zip(z, np.squeeze(X))), **dict(zip(J_decision_variables, alpha))}
        dJdz_val = [dJdz_poly.Evaluate(dict_val)]
        for i in range(1, nz):
            dJdz_poly = Polynomial(dJdz[i], z)
            calc_basis_dJdz = construct_monomial_basis_from_polynomial(dJdz_poly, nJ, z)
            dphi_dx = np.expand_dims(calc_basis_dJdz(X), axis=1)
            dPhi_dx = np.concatenate((dPhi_dx, dphi_dx), axis=1)
            dJdz_val.append(dJdz_poly.Evaluate(dict_val))
        
        dJdz_basis_val = np.squeeze(dPhi_dx @ alpha)
        dJdz_val = np.array(dJdz_val)
        np.testing.assert_allclose(dJdz_basis_val, dJdz_val)


if __name__ == '__main__':
    nz = 4
    deg = 6
    unittest.main()