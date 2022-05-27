import unittest
import numpy as np
import torch
from cartpole_rks_nn import cartpole_setup
from cartpole_swingup_fvi import cartpole_batch_setup
from pydrake.all import (DiagramBuilder, AddMultibodyPlantSceneGraph, Parser, MultibodyPlantConfig)
from underactuated import FindResource

class TestAcrobot(unittest.TestCase):
    def test_time_derivatives(self):
        x_tensor = torch.tensor(x, dtype=dtype).unsqueeze(0)
        f1_val = params_dict["f1"](x_tensor)
        f2_val = params_dict["f2"](x_tensor)
        f_val = f1_val.squeeze().detach().numpy() + np.squeeze(f2_val.detach().numpy() @ u)

        np.testing.assert_allclose(f_val, f.CopyToVector())
    
    def test_cartpole_batch(self):
        X = np.expand_dims(x, axis=0)

        params_dict = cartpole_batch_setup()
        T_val = params_dict["T"](X)
        f1_val = params_dict["f1"](X, T_val)
        f2_val = params_dict["f2"](X, T_val)
        f_val = np.squeeze(f1_val) + np.squeeze(f2_val @ u)

        np.testing.assert_allclose(f_val, np.squeeze(T_val)@f.CopyToVector())


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
    mc = 10
    mp = 1
    l = .5
    g = 9.81
    dtype = torch.float64

    u = np.array([1])
    plant.get_actuation_input_port().FixValue(context, u)
    f = plant.AllocateTimeDerivatives()
    plant.CalcTimeDerivatives(context, f)

    unittest.main()