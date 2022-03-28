import bezier_operation 
import numpy as np
import unittest

class TestBezier(unittest.TestCase):
    def test_power_conversion(self):
        pg = np.zeros([2,2])
        pg[0, 0] = -1
        pg[1, 0] = 1
        pg[0, 1] = 1
        Pg = bezier_operation.power_to_bernstein_poly(pg)
        G11 = np.array([[-1, 0], [0, 1]])
        np.testing.assert_allclose(Pg, G11)

        pf = np.zeros([3,2])
        pf[0, 0] = 3
        pf[0, 1] = 1
        pf[2, 1] = 1
        Pf = bezier_operation.power_to_bernstein_poly(pf)
        F21 = np.array([[3, 4], [3, 4], [3, 5]])
        np.testing.assert_allclose(Pf, F21)
    
    def test_bernstein_mul(self):
        p1 = np.array([1, 2, 3])
        p2 = np.array([9, 5, 1])
        # np poly accepts coefficients from hight to low degree
        power_mul = np.polymul(np.flip(p1), np.flip(p2))

        p1_bern = bezier_operation.power_to_bernstein_poly(p1)
        p2_bern = bezier_operation.power_to_bernstein_poly(p2)
        power_mul_bern = bezier_operation.power_to_bernstein_poly(np.flip(power_mul))

        mul = bezier_operation.bernstein_mul(p1_bern, p2_bern)
        np.testing.assert_allclose(mul, power_mul_bern)

    def test_bernstein_add(self):
        p1 = np.array([1, 2, 3, 4])
        p2 = np.zeros(3)
        # p2 = np.array([9, 5, 1])
        power_add = np.polyadd(np.flip(p1), np.flip(p2))

        p1_bern = bezier_operation.power_to_bernstein_poly(p1)
        p2_bern = bezier_operation.power_to_bernstein_poly(p2)
        power_add_bern = bezier_operation.power_to_bernstein_poly(np.flip(power_add))

        add = bezier_operation.bernstein_add(p1_bern, p2_bern)
        np.testing.assert_allclose(add, power_add_bern)

    def test_bernstein_derivative(self):
        p = np.array([9, 5, 1])
        p_der = np.polyder(np.flip(p))
        p_bern = bezier_operation.power_to_bernstein_poly(p)
        p_der_bern = bezier_operation.power_to_bernstein_poly(np.flip(p_der))

        der = bezier_operation.bernstein_derivative(p_bern)
        np.testing.assert_allclose(p_der_bern, der[0])

    def test_bernstein_definite_integral(self):
        p = np.array([9, 5, 1])
        p_int = np.polyint(np.flip(p))
        p_bern = bezier_operation.power_to_bernstein_poly(p)
        p_int_bern = np.polyval(p_int,0.5)

        integral = bezier_operation.bernstein_definite_integral(p_bern, 0.5)
        np.testing.assert_allclose(p_int_bern, integral)

    def test_bernstein_integral(self):
        p = np.array([9, 5, 1])
        p_int = np.polyint(np.flip(p))
        p_bern = bezier_operation.power_to_bernstein_poly(p)
        lo = 0.2
        hi = 0.8
        p_int_bern = np.polyval(p_int, hi) - np.polyval(p_int, lo)

        integral = bezier_operation.bernstein_integral(p_bern, lo, hi)
        np.testing.assert_allclose(p_int_bern, integral)


if __name__ == '__main__':
    unittest.main()