import unittest
import numpy as np
from acrobot_swingup_fvi import acrobot_setup
from pydrake.all import (Simulator, ContinuousState)
from pydrake.examples.acrobot import (AcrobotPlant, AcrobotInput) 

class TestAcrobot(unittest.TestCase):
    def test_Minv(self):
        Minv_val = params_dict["Minv"](x)
        M = acrobot.MassMatrix(context)
        np.testing.assert_allclose(Minv_val, np.linalg.inv(M))

    def test_time_derivatives(self):
        Minv_val = params_dict["Minv"](x)
        T_val = params_dict["T"](params_dict["x2z"](x))
        # Remember to remove dependence on T
        u = 1
        f_val = params_dict["f"](x, Minv_val, np.array([u]), T_val)
        input = AcrobotInput()
        input.set_tau(u)
        acrobot.GetInputPort("elbow_torque").FixValue(context, input)
        f = acrobot.AllocateTimeDerivatives()
        acrobot.CalcTimeDerivatives(context, f)
        np.testing.assert_allclose(f_val.astype(float), T_val@f.CopyToVector())

if __name__ == '__main__':
    acrobot = AcrobotPlant()
    context = acrobot.CreateDefaultContext()
    acrobot_params = acrobot.get_parameters(context)
    acrobot_params.set_b1(0)
    acrobot_params.set_b2(0)
    x = np.random.random(4)
    state = context.get_mutable_continuous_state_vector().SetFromVector(x)
    params_dict = acrobot_setup()
    unittest.main()