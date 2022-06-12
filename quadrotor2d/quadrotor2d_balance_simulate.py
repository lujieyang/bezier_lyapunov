# %%
import sys
sys.path.append("../../underactuated")

# %%
import numpy as np
import pickle
from IPython.display import HTML, display
from utils import calc_u_opt, reconstruct_polynomial_from_dict
from quadrotor2d_sos import quadrotor2d_sos_lower_bound
from pydrake.all import (DiagramBuilder, Simulator, WrapToSystem, LeafSystem,
                         BasicVector, MathematicalProgram)
from underactuated.quadrotor2d import Quadrotor2D, Quadrotor2DVisualizer
import meshcat

# %%
class Controller(LeafSystem):
    def __init__(self, J_star, z):
        LeafSystem.__init__(self)
        self.x_dim = 6
        self.u_dim = 2
        self.nz,  _, self.f2, self.x2z, self.Rinv = quadrotor2d_sos_lower_bound(2, test=True)
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
        f2_val = self.f2(z_val, float)
        dJdz_val = np.zeros(self.nz)
        for n in range(self.nz): 
            dJdz_val[n] = self.dJdz[n].Evaluate(dict(zip(z, z_val)))
        u_opt = calc_u_opt(dJdz_val, f2_val, self.Rinv)
        y[:]  = u_opt
        # print(u_opt)

# %%
def simulate(J_star, z):
    # Animate the resulting policy.
    builder = DiagramBuilder()
    plant = builder.AddSystem(Quadrotor2D())
    controller = builder.AddSystem(Controller(J_star, z))
    builder.Connect(controller.get_output_port(0), plant.get_input_port(0))
    builder.Connect(plant.get_output_port(0), controller.get_input_port(0))

    # Setup visualization
    visualizer = builder.AddSystem(Quadrotor2DVisualizer(show=False))
    builder.Connect(plant.get_output_port(0), visualizer.get_input_port(0))

    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    context.SetContinuousState([1, 0, 0, 0, 0, 0])
    visualizer.start_recording()
    simulator.AdvanceTo(5)
    ani = visualizer.get_recording_as_animation()
    display(HTML(ani.to_jshtml()))

# %%
def set_orthographic_camera_xy(vis: meshcat.Visualizer) -> None:
    # use orthographic camera, show XY plane.
    camera = meshcat.geometry.OrthographicCamera(
        left=-1, right=0.5, bottom=-0.5, top=1, near=-1000, far=1000)
    vis['/Cameras/default/rotated'].set_object(camera)

# %%
deg = 2
prog = MathematicalProgram()
z = prog.NewIndeterminates(7, "z")
with open("quadrotor2d/data/J_lower_bound_deg_{}.pkl".format(deg), "rb") as input_file:
    C = pickle.load(input_file)
J_star = reconstruct_polynomial_from_dict(C, z)

# %%
simulate(J_star, z)

# %%



