import unittest
import numpy as np
from acrobot_swingup_fvi import acrobot_setup
from acrobot_sos_swingup import acrobot_sos_iterative_upper_bound_implicit_dynamics, acrobot_sos_lower_bound_implicit_dynamics, acrobot_sos_upper_bound, acrobot_sos_iterative_upper_bound
from pydrake.all import (Simulator, ContinuousState)
from pydrake.examples.acrobot import (AcrobotPlant, AcrobotInput) 

class TestAcrobot(unittest.TestCase):
    def test_Minv(self):
        Minv_val = params_dict["Minv"](x)
        np.testing.assert_allclose(Minv_val, np.linalg.inv(M_drake))

    def test_time_derivatives(self):
        Minv_val = params_dict["Minv"](x)
        T_val = params_dict["T"](params_dict["x2z"](x))
        # Remember to remove dependence on T
        f_val = params_dict["f"](x, Minv_val, np.array([u]), T_val)
        np.testing.assert_allclose(f_val.astype(float), T_val@f.CopyToVector())

    def test_xdot_lower_bound(self):
        M0, F, M = acrobot_sos_lower_bound_implicit_dynamics(2, test=True)
        x2z = params_dict["x2z"]
        z = x2z(x)
        np.testing.assert_allclose(M(z), M_drake)

        M0_val = M0(z, float)
        F_val = F(z, np.array([u]), float)
        np.testing.assert_allclose(np.linalg.inv(M0_val)@F_val, f.CopyToVector())

    def test_xdot_upper_bound(self):
        nz, f_func, f2, T, z0 = acrobot_sos_upper_bound(2, test=True)
        x2z = params_dict["x2z"]
        z = x2z(x)
        f_val, det_M = f_func(z, np.array([u]), float)
        np.testing.assert_allclose(f_val/det_M, T(z, float)@f.CopyToVector())

    def test_xdot_iterative_upper_bound(self):
        nz, f_func, f2, T, z0, Rinv = acrobot_sos_iterative_upper_bound(2, test=True)
        x2z = params_dict["x2z"]
        z = x2z(x)
        u = 1
        _, det_M = f_func(z, np.array([u]), float)
        f2_bar, _ = f2(z, float)
        u_bar = np.array([- np.ones(nz).dot(np.squeeze(f2_bar))])
        f_bar, _ = f_func(z, u_bar, dtype=float, u_denominator=det_M)
        u = u_bar/det_M
        input.set_tau(u)
        acrobot.GetInputPort("elbow_torque").FixValue(context, input)
        f = acrobot.AllocateTimeDerivatives()
        acrobot.CalcTimeDerivatives(context, f)
        np.testing.assert_allclose(f_bar/det_M**2, T(z, float)@f.CopyToVector())

    def test_xdot_iterative_upper_bound_implicit_dynamics(self):
        nz, F_bar, M_bar, f2, T, z0, Rinv = acrobot_sos_iterative_upper_bound_implicit_dynamics(2, test=True)
        x2z = params_dict["x2z"]
        z = x2z(x)
        u = 1

        f2_bar, u_denominator = f2(z, float)
        u_bar = np.array([- np.ones(nz).dot(np.squeeze(f2_bar))])
        F_bar_val = F_bar(z, u_bar, dtype=float, u_denominator=u_denominator)
        M_bar_val = M_bar(z, dtype=float)
        u = u_bar/u_denominator
        input.set_tau(u)
        acrobot.GetInputPort("elbow_torque").FixValue(context, input)
        f = acrobot.AllocateTimeDerivatives()
        acrobot.CalcTimeDerivatives(context, f)
        np.testing.assert_allclose(np.linalg.inv(M_bar_val)@F_bar_val/u_denominator, T(z, float)@f.CopyToVector())

if __name__ == '__main__':
    acrobot = AcrobotPlant()
    context = acrobot.CreateDefaultContext()
    acrobot_params = acrobot.get_parameters(context)
    acrobot_params.set_b1(0)
    acrobot_params.set_b2(0)
    x = np.random.random(4)
    state = context.get_mutable_continuous_state_vector().SetFromVector(x)
    params_dict = acrobot_setup()
    u = 1
    input = AcrobotInput()
    input.set_tau(u)
    acrobot.GetInputPort("elbow_torque").FixValue(context, input)
    f = acrobot.AllocateTimeDerivatives()
    acrobot.CalcTimeDerivatives(context, f)
    M_drake = acrobot.MassMatrix(context)
    unittest.main()