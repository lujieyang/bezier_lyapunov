# %%
import sys
sys.path.append("../../underactuated")

# %%
import numpy as np
import pickle
from utils import calc_u_opt, reconstruct_polynomial_from_dict
from acrobot_swingup_fvi import acrobot_setup
from pydrake.examples.acrobot import (AcrobotPlant, AcrobotGeometry)
from pydrake.all import (DiagramBuilder, Simulator, WrapToSystem, LeafSystem,
                         BasicVector, MathematicalProgram, SceneGraph, MeshcatVisualizerCpp,
                         Expression, ConnectMeshcatVisualizer)
from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
import meshcat

# %%
class Controller(LeafSystem):
    def __init__(self, J_star, z, plant, params_dict):
        LeafSystem.__init__(self)
        self.plant = plant
        self.context = plant.CreateDefaultContext()
        self.x_dim = 4
        self.u_dim = 1
        self.nz = params_dict["nz"]
        self.x2z = params_dict["x2z"]
        self.Minv = params_dict["Minv"]
        self.T = params_dict["T"]
        self.f = params_dict["f"]
        self.l = params_dict["l"]
        self.f2 = params_dict["f2"]
        self.dJdz = J_star.Jacobian(z)
        self.z = z
        
        self.state_input_port = self.DeclareVectorInputPort(
            "state", BasicVector(self.x_dim))
        self.policy_output_port = self.DeclareVectorOutputPort(
            "policy", BasicVector(self.u_dim), self.CalculateController)
        
    def CalculateController(self, context, output):
        x = self.state_input_port.Eval(context)
        z_val = self.x2z(x)
        y = output.get_mutable_value()
        Minv_val = self.Minv(x)
        T_val = self.T(z_val)
        f2_val = self.f2(Minv_val, T_val)
        dJdz_val = np.zeros(self.nz)
        for n in range(self.nz): 
            dJdz_val[n] = self.dJdz[n].Evaluate(dict(zip(z, z_val)))
        u_opt = calc_u_opt(dJdz_val, f2_val, params_dict["Rinv"])
        y[:]  = u_opt
        print(u_opt)

# %%
def simulate(J_star, z, params_dict):
    # Animate the resulting policy.
    builder = DiagramBuilder()
    plant = AcrobotPlant()
    acrobot = builder.AddSystem(plant)
    wrap = builder.AddSystem(WrapToSystem(4))
    wrap.set_interval(0, 0, 2*np.pi)
    wrap.set_interval(1, 0, 2*np.pi)
    builder.Connect(acrobot.get_output_port(0), wrap.get_input_port(0))
    vi_policy = Controller(J_star, z, acrobot, params_dict)
    builder.AddSystem(vi_policy)
    builder.Connect(wrap.get_output_port(0), vi_policy.get_input_port(0))
    builder.Connect(vi_policy.get_output_port(0),
                    acrobot.get_input_port(0))

    # Setup visualization
    scene_graph = builder.AddSystem(SceneGraph())
    AcrobotGeometry.AddToBuilder(builder, acrobot.get_output_port(0),
                                scene_graph)

    proc, zmq_url, web_url = start_zmq_server_as_subprocess(server_args=[])
    viz = ConnectMeshcatVisualizer(builder, scene_graph, zmq_url=zmq_url)
    # set_orthographic_camera_xy(viz.vis)
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    acrobot_context = acrobot.GetMyMutableContextFromRoot(context)
    acrobot_params = plant.get_parameters(acrobot_context)
    acrobot_params.set_b1(0)
    acrobot_params.set_b2(0)
    context.SetContinuousState([np.pi-0.1, 0, 0, 0])
    viz.start_recording()
    simulator.AdvanceTo(1)
    viz.publish_recording()

# %%
def set_orthographic_camera_xy(vis: meshcat.Visualizer) -> None:
    # use orthographic camera, show XY plane.
    camera = meshcat.geometry.OrthographicCamera(
        left=-1, right=0.5, bottom=-0.5, top=1, near=-1000, far=1000)
    vis['/Cameras/default/rotated'].set_object(camera)

# %%
params_dict = acrobot_setup()
poly_deg = 4
n_mesh = 11
prog = MathematicalProgram()
z = prog.NewIndeterminates(params_dict["nz"], "z")
with open("acrobot/data/J_{}_{}.pkl".format(poly_deg, n_mesh), "rb") as input_file:
    C = pickle.load(input_file)
J_star = reconstruct_polynomial_from_dict(C, z)

# %%
simulate(J_star, z, params_dict)

# %%



