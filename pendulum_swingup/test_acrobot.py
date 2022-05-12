import unittest
import numpy as np
from acrobot_swingup_fvi import acrobot_setup
from pydrake.all import (Simulator, ContinuousState)
from pydrake.examples.acrobot import (AcrobotPlant) 

class TestAcrobot(unittest.TestCase):
    def test_Minv(self):
        Minv_val = params_dict["Minv"](x)
        M = acrobot.MassMatrix(context)
        np.testing.assert_allclose(Minv_val, np.linalg.inv(M))

    def test_time_derivatives(self):
        Minv_val = params_dict["Minv"](x)
        T_val = params_dict["T"](params_dict["x2z"](x))
        # Remember to remove dependence on T
        f_val = params_dict["f"](x, Minv_val, np.array([0]), T_val)
        f = acrobot.AllocateTimeDerivatives()
        acrobot.CalcTimeDerivatives(context, f)
        np.testing.assert_allclose(f_val, f.CopyToVector())

if __name__ == '__main__':
    acrobot = AcrobotPlant()
    simulator = Simulator(acrobot)
    context = simulator.get_mutable_context()
    acrobot_params = acrobot.get_parameters(context)
    acrobot_params.set_b1(0)
    acrobot_params.set_b2(0)
    state = acrobot.get_state(context)
    state.set_theta1(1.)
    state.set_theta1dot(2.)
    state.set_theta2(3.)
    state.set_theta2dot(4.)
    params_dict = acrobot_setup()
    # x = np.array([1, 2, 3, 4])
    x = np.zeros(4)
    unittest.main()