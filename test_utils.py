import unittest
import numpy as np
from utils import matrix_adjoint, matrix_det

class TestUtils(unittest.TestCase):
    def test_matrix_det(self):
        X = np.random.rand(4,4)
        det_np = np.linalg.det(X)
        det = matrix_det(X)
        np.testing.assert_almost_equal(det, det_np)
    
    def test_matrix_adjoint(self):
        X = np.random.rand(2,2)
        inv_np = np.linalg.inv(X)
        inv = matrix_adjoint(X)/matrix_det(X)
        np.testing.assert_allclose(inv, inv_np)

if __name__ == '__main__':
    unittest.main()