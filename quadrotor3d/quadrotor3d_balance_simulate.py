import numpy as np
import pickle
from utils import calc_u_opt, reconstruct_polynomial_from_dict
from pydrake.all import (MathematicalProgram, DiagramBuilder, SceneGraph, LeafSystem,
                         MeshcatVisualizerCpp, Simulator, StartMeshcat, RollPitchYaw, BasicVector)
from quadrotor3d.quadrotor3d_sos import quadrotor3d_sos_upper_bound, plot_value_function
from pydrake.examples.quadrotor import (QuadrotorGeometry, QuadrotorPlant)

import matplotlib.pyplot as plt
from matplotlib import cm

meshcat = StartMeshcat()

def ToTrigState(x_original):
    rpy = RollPitchYaw(x_original[3:6])
    quaternion = rpy.ToQuaternion()

    x_trig = np.zeros(13)
    x_trig[0] = quaternion.w() - 1
    x_trig[1] = quaternion.x()
    x_trig[2] = quaternion.y()
    x_trig[3] = quaternion.z()
    x_trig[4:7] = x_original[:3]
    x_trig[7:10] = x_original[6:9]
    x_trig[10:] = rpy.CalcAngularVelocityInChildFromRpyDt(x_original[-3:])
    return x_trig

class Controller(LeafSystem):
    def __init__(self, J_star, z):
        LeafSystem.__init__(self)
        quadrotor = QuadrotorPlant()

        self.u0 = quadrotor.m() * quadrotor.g() / 4. * np.ones(4)
        self.nu = 4
        self.nz, _, self.f2, self.z_max, self.z_min = quadrotor3d_sos_upper_bound(2, test=True)
        self.dJdz = J_star.Jacobian(z)
        Rinv = np.linalg.inv(np.eye(4))
        f2_val = self.f2(z)
        self.u_star = - .5 * Rinv.dot(f2_val.T).dot(self.dJdz.T)
        self.z = z
        
        self.state_input_port = self.DeclareVectorInputPort(
            "state", BasicVector(12))
        self.policy_output_port = self.DeclareVectorOutputPort(
            "policy", BasicVector(4), self.CalculateController)
        
    def CalculateController(self, context, output):
        x = self.state_input_port.Eval(context)
        z_val = ToTrigState(x)
        y = output.get_mutable_value()
        u_opt = np.zeros(self.nu)
        for i in range(self.nu):
            u_opt[i] = self.u_star[i].Evaluate(dict(zip(z, z_val)))
        y[:]  = u_opt + self.u0  # CAUTION: Add u_equilibrium
        # print("y:", y)
        # print("x:", x)

def simulate(J_star, z):
    builder = DiagramBuilder()

    plant = builder.AddSystem(QuadrotorPlant())

    controller = builder.AddSystem(Controller(J_star, z))
    builder.Connect(controller.get_output_port(0), plant.get_input_port(0))
    builder.Connect(plant.get_output_port(0), controller.get_input_port(0))

    # Set up visualization in MeshCat
    scene_graph = builder.AddSystem(SceneGraph())
    QuadrotorGeometry.AddToBuilder(builder, plant.get_output_port(0), scene_graph)
    meshcat.Delete()
    meshcat.ResetRenderMode()
    meshcat.SetProperty('/Background','visible',False)
    viz = MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat)
    # end setup for visualization

    diagram = builder.Build()

    # Set up a simulator to run this diagram
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    context = simulator.get_mutable_context()

    # Simulate
    context.SetTime(0.)
    x0 = np.ones(12)
    x0[3:6] = np.array([np.pi, 0.4 * np.pi, np.pi])
    context.SetContinuousState(x0)
    simulator.Initialize()
    viz.StartRecording()
    simulator.AdvanceTo(10.0)
    viz.StopRecording()
    viz.PublishRecording()
    print(context.get_continuous_state().CopyToVector())

if __name__ == '__main__':
    deg = 2
    prog = MathematicalProgram()
    z = prog.NewIndeterminates(13, "z")
    with open("quadrotor3d/data/J_upper_bound_deg_{}.pkl".format(deg), "rb") as input_file:
        C = pickle.load(input_file)
    J_star = reconstruct_polynomial_from_dict(C, z)

    simulate(J_star, z)