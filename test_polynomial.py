import numpy as np
import unittest
from pydrake.all import (Variable)
from polynomial_fvi import *


class TestPolynomial(unittest.TestCase):
    def test_chebyshev(self):
        x = Variable("x")
        U4 = 16 * x**4 - 12* x**2 + 1
        self.assertTrue(U4.Evaluate({x:4}) == chebyshev_polynomial(x, 4, 0).Evaluate({x:4}))
        
    def test_legendre(self):
        x = Variable("x")
        P4 = (35 * x**4 - 30 * x**2 + 3)/8
        self.assertTrue(P4.Evaluate({x:4}) == legendre_polynomial(x, 4, 0).Evaluate({x:4}))

if __name__ == '__main__':
    unittest.main()