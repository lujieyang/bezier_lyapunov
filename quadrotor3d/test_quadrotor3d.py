import unittest
import numpy as np
from quadrotor3d_sos import quadrotor3d_sos_upper_bound, quadrotor3d_trig_constrained_lqr
from pydrake.examples.quadrotor_trig import (QuadrotorTrigPlant)

class TestAcrobot(unittest.TestCase):
    def test_xdot(self):
        nz, f, f2, z_max, z_min = quadrotor3d_sos_upper_bound(2, test=True)
        z = x

        f_drake_val = f_drake.CopyToVector()
        f_val = f(z, u, float)
        np.testing.assert_allclose(f_val, f_drake_val)

    # def test_constrained_lqr(self):
    #     K, S = quadrotor3d_trig_constrained_lqr()
    #     K_drake, S_drake = quadrotor.SynthesizeTrigLqr()

    #     np.testing.assert_allclose(K, K_drake)
    #     np.testing.assert_allclose(S, S_drake)

    def test_f2(self):
        nz, f, f2, z_max, z_min = quadrotor3d_sos_upper_bound(2, test=True)
        f1_val = f(x, np.zeros(4), float)
        f2_val = f2(x)
        f_val = f(x, u, float)
        f_from_f2 = f1_val + f2_val @ u

        np.testing.assert_almost_equal(f_val, f_from_f2)

if __name__ == '__main__':
    quadrotor = QuadrotorTrigPlant()
    context = quadrotor.CreateDefaultContext()
    x = np.zeros(13)
    x[:4] = np.array([-1, 0.6, 0.8, 0])
    x[4:] = np.random.random(9)
    state = context.get_mutable_continuous_state_vector().SetFromVector(x)
    u = np.random.random(4)
    quadrotor.get_input_port().FixValue(context, u)
    f_drake = quadrotor.AllocateTimeDerivatives()
    quadrotor.CalcTimeDerivatives(context, f_drake)
    unittest.main()