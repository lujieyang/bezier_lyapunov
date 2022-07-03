import numpy as np
from pydrake.all import (LinearQuadraticRegulator, DiagramBuilder, SceneGraph, WrapToSystem, LeafSystem,
                         MeshcatVisualizerCpp, Simulator, StartMeshcat, RollPitchYaw, Linearize, BasicVector)
from quadrotor3d.quadrotor3d_sos import quadrotor3d_trig_constrained_lqr, quadrotor3d_sos_upper_bound
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

class ConstrainedLQR(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        m = 0.775
        g = 9.81
        self.nz = 13
        self.nu = 4
        self.u0 = m * g / 4. * np.ones(self.nu)
        self.z0 = np.zeros(self.nz)
        self.K, _ = quadrotor3d_trig_constrained_lqr()
        self.state_input_port = self.DeclareVectorInputPort(
            "state", BasicVector(12))
        self.policy_output_port = self.DeclareVectorOutputPort(
            "policy", BasicVector(4), self.CalculateController)

    def ToTrigState(self, x_original):
        rpy = RollPitchYaw(x_original[3:6])
        quaternion = rpy.ToQuaternion()

        x_trig = np.zeros(self.nz)
        x_trig[0] = quaternion.w() - 1
        x_trig[1] = quaternion.x()
        x_trig[2] = quaternion.y()
        x_trig[3] = quaternion.z()
        x_trig[4:7] = x_original[:3]
        x_trig[7:10] = x_original[6:9]
        x_trig[10:] = rpy.CalcAngularVelocityInChildFromRpyDt(x_original[-3:])
        return x_trig
        
    def CalculateController(self, context, output):
        x = self.state_input_port.Eval(context)
        y = output.get_mutable_value()
        z = ToTrigState(x)
        y[:]  = -self.K@(z-self.z0) + self.u0

def quadrotor_balancing_example():
    builder = DiagramBuilder()

    plant = builder.AddSystem(QuadrotorPlant())

    controller = builder.AddSystem(ConstrainedLQR())
    builder.Connect(controller.get_output_port(0), plant.get_input_port(0))
    builder.Connect(plant.get_output_port(0), controller.get_input_port(0))

    # Set up visualization in MeshCat
    scene_graph = builder.AddSystem(SceneGraph())
    QuadrotorGeometry.AddToBuilder(builder, plant.get_output_port(0), scene_graph)
    meshcat.Delete()
    meshcat.ResetRenderMode()
    meshcat.SetProperty('/Background','visible',False)
    MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat)
    # end setup for visualization

    diagram = builder.Build()

    # Set up a simulator to run this diagram
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    context = simulator.get_mutable_context()

    # Simulate
    for i in range(5):
        context.SetTime(0.)
        context.SetContinuousState(0.5*np.random.randn(12,))
        simulator.Initialize()
        simulator.AdvanceTo(8.0)
        print(context.get_continuous_state().CopyToVector())


def plot_lqr_cost_to_go(z_min, z_max, plot_states="xy", u_index=0):
    quadrotor = QuadrotorPlant()
    context = quadrotor.CreateDefaultContext()

    u0 = quadrotor.m() * quadrotor.g() / 4. * np.ones(4)
    quadrotor.get_input_port(0).FixValue(context, u0)

    z0 = np.zeros(13)
    K, S = quadrotor3d_trig_constrained_lqr()

    zero_vector = np.zeros([51*51, 1])
    qwxyz = np.zeros([51*51, 4])
    xyzdot_w = np.zeros([51*51, 6])
    if plot_states == "xz":
        y_limit_idx = 6
        X, Z1 = np.meshgrid(np.linspace(z_min[4], z_max[4], 51),
                        np.linspace(z_min[6], z_max[6], 51))
        Z = np.hstack((qwxyz, X.flatten().reshape(51*51, 1), zero_vector, Z1.flatten().reshape(51*51, 1), xyzdot_w))
        ylabel="z"
    elif plot_states == "xy":
        y_limit_idx = 5
        X, Y = np.meshgrid(np.linspace(z_min[4], z_max[4], 51),
                        np.linspace(z_min[5], z_max[5], 51))
        Z = np.hstack((qwxyz, X.flatten().reshape(51*51, 1), Y.flatten().reshape(51*51, 1), zero_vector, xyzdot_w))
        ylabel="y"

    J = np.zeros(Z.shape[0])
    U = np.zeros(Z.shape[0])

    for i in range(Z.shape[0]):
        z = np.squeeze(Z[i]) - z0
        J[i] = (z.dot(S)).dot(z)
        U[i] = -K.dot(z)[u_index] + u0[u_index]

    fig = plt.figure(figsize=(9, 4))
    ax = fig.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.set_title("Cost-to-Go")
    im = ax.imshow(J.reshape(X.shape),
            cmap=cm.jet, aspect='auto',
            extent=(z_min[4], z_max[4], z_max[y_limit_idx], z_min[y_limit_idx]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.savefig("quadrotor3d/figures/trig_constrained_lqr_cost_to_go_{}.png".format(plot_states))

    fig = plt.figure(figsize=(9, 4))
    ax = fig.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.set_title("Policy")
    im = ax.imshow(U.reshape(X.shape),
            cmap=cm.jet, aspect='auto',
            extent=(z_min[4], z_max[4], z_max[y_limit_idx], z_min[y_limit_idx]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.savefig("quadrotor3d/figures/trig_constrained_lqr_policy_{}_u_{}.png".format(plot_states, u_index+1))


if __name__ == '__main__':
    # quadrotor_balancing_example()
    _, _, _, z_max, z_min = quadrotor3d_sos_upper_bound(2, test=True)
    plot_lqr_cost_to_go(z_min, z_max, plot_states="xz")