# %%
import sys
sys.path.append("../../underactuated")

# %%
import numpy as np
import torch
from torch.autograd import grad
from cartpole_rks_nn import cartpole_setup, setup_nn
from pydrake.all import (DiagramBuilder, Simulator, WrapToSystem, LeafSystem,
                         BasicVector, Parser, SceneGraph, AddMultibodyPlantSceneGraph,
                         MathematicalProgram, ConnectMeshcatVisualizer)
from underactuated import FindResource

from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
import meshcat

# %%
class Controller(LeafSystem):
    def __init__(self, alpha, sinks, params_dict):
        LeafSystem.__init__(self)
        self.x_dim = 4
        self.u_dim = 1
        self.alpha = alpha
        self.sinks = sinks
        self.f2 = params_dict["f2"]
        self.Rinv = params_dict['Rinv'].squeeze()
        self.dtype = torch.float64
        self.K = len(sinks)
        
        self.state_input_port = self.DeclareVectorInputPort(
            "state", BasicVector(self.x_dim))
        self.policy_output_port = self.DeclareVectorOutputPort(
            "policy", BasicVector(self.u_dim), self.CalculateController)

    def CalculateController(self, context, output):
        x = self.state_input_port.Eval(context)
        x_tensor = torch.tensor(x, dtype=self.dtype)
        x_tensor.requires_grad = True
        basis = sinks[0](x_tensor)
        dPhi_dx = grad(basis, x_tensor, grad_outputs=torch.ones_like(basis))[0]
        for k in range(1, K):
            b = sinks[k](x_tensor)
            dphi_dx = grad(b, x_tensor, grad_outputs=torch.ones_like(b))[0]
            basis = torch.hstack((basis, b))
            dPhi_dx = torch.vstack((dPhi_dx, dphi_dx))
        dJdx = dPhi_dx.T @ self.alpha
        f2_val = self.f2(x_tensor.unsqueeze(0)).squeeze()
        u_opt = -self.Rinv*f2_val.T.dot(dJdx)/2

        y = output.get_mutable_value()
        y[:]  = np.clip(u_opt.detach().numpy(), -300, 300)

# %%
def simulate(alpha, sinks, params_dict):
    # Animate the resulting policy.
    builder = DiagramBuilder()
    cartpole, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    file_name = FindResource("models/cartpole.urdf")
    Parser(cartpole).AddModelFromFile(file_name)
    cartpole.Finalize()

    wrap = builder.AddSystem(WrapToSystem(4))
    wrap.set_interval(1, 0, 2*np.pi)
    builder.Connect(cartpole.get_state_output_port(), wrap.get_input_port(0))
    vi_policy = Controller(alpha, sinks, params_dict)
    builder.AddSystem(vi_policy)
    builder.Connect(wrap.get_output_port(0), vi_policy.get_input_port(0))
    builder.Connect(vi_policy.get_output_port(0),
                    cartpole.get_actuation_input_port())

    proc, zmq_url, web_url = start_zmq_server_as_subprocess(server_args=[])
    viz = ConnectMeshcatVisualizer(builder, scene_graph, zmq_url=zmq_url)
    # set_orthographic_camera_xy(viz.vis)
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    context.SetContinuousState([0, np.pi-0.1, 0, 0])
    viz.start_recording()
    simulator.AdvanceTo(0.1)
    viz.publish_recording()

# %%
def set_orthographic_camera_xy(vis: meshcat.Visualizer) -> None:
    # use orthographic camera, show XY plane.
    camera = meshcat.geometry.OrthographicCamera(
        left=-1, right=0.5, bottom=-0.5, top=1, near=-1000, far=1000)
    vis['/Cameras/default/rotated'].set_object(camera)

# %%
params_dict = cartpole_setup()
K = 200
n_mesh = 10
h_layer = 16
activation_type = "sigmoid"
sink_dict = torch.load("cartpole/data/rks_nn/{}/alpha_{}_sink_{}_mesh_{}_hidden.pth".format(activation_type, K, n_mesh, h_layer))
alpha = sink_dict["alpha"]
sinks = []
for k in range(K):
    sinks.append(setup_nn((params_dict["nx"], h_layer, h_layer, 1)))
    sinks[k].load_state_dict(sink_dict[k])
    sinks[k].eval()
# %%
simulate(alpha, sinks, params_dict)

# %%



