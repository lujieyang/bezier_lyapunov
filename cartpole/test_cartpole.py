import unittest
import numpy as np
from cartpole_swingup_fvi import cartpole_setup
from pydrake.all import (DiagramBuilder, AddMultibodyPlantSceneGraph, Parser, MultibodyPlantConfig)
from underactuated import FindResource

class TestAcrobot(unittest.TestCase):
    def test_M(self):
        mc = 10
        mp = 1
        l = .5
        g = 9.81
        c = np.cos(x[1] + np.pi)  # Drake calculation and underactuated notes off by pi
        M = np.array([[mc+mp, mp*l*c],
                     [mp*l*c, mp*l**2]])
        M_drake = plant.CalcMassMatrix(context)
        np.testing.assert_allclose(M, M_drake)

    def test_time_derivatives(self):
        T_val = params_dict["T"](params_dict["x2z"](x))
        # Remember to remove dependence on T
        u = np.array([1])
        f_val = params_dict["f"](x, u, T_val)
        plant.get_actuation_input_port().FixValue(context, u)
        f = plant.AllocateTimeDerivatives()
        plant.CalcTimeDerivatives(context, f)
        # np.testing.assert_allclose(f_val.astype(float), T_val@f.CopyToVector())

if __name__ == '__main__':
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    file_name = FindResource("models/cartpole.urdf")
    Parser(plant).AddModelFromFile(file_name)
    plant.Finalize()
    x = np.random.random(4)
    context = plant.CreateDefaultContext()
    context.SetContinuousState(x)
    params_dict = cartpole_setup()
    unittest.main()