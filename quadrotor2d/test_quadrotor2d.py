import unittest
import numpy as np
from quadrotor2d_sos import quadrotor2d_sos_lower_bound
from pydrake.all import (Simulator, ContinuousState)
from underactuated.quadrotor2d import Quadrotor2D

class TestAcrobot(unittest.TestCase):
    def test_xdot(self):
        f, f2, x2z, Rinv = quadrotor2d_sos_lower_bound(2, test=True)
        z = x2z(x)

        f_drake_val = f_drake.CopyToVector()
        f_drakez = np.zeros(7)
        f_drakez[:2] = f_drake_val[:2]
        f_drakez[-3:] = f_drake_val[-3:]
        f_drakez[2] = x[-1] * np.cos(x[2])
        f_drakez[3] = -x[-1] * np.sin(x[2])
        f_val = f(z, u, float)
        np.testing.assert_allclose(f_val, f_drakez)

if __name__ == '__main__':
    quadrotor = Quadrotor2D()
    context = quadrotor.CreateDefaultContext()
    x = np.random.random(6)
    state = context.get_mutable_continuous_state_vector().SetFromVector(x)
    u = np.random.random(2)
    quadrotor.get_input_port().FixValue(context, u)
    f_drake = quadrotor.AllocateTimeDerivatives()
    quadrotor.CalcTimeDerivatives(context, f_drake)
    unittest.main()