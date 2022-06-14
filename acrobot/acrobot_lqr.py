import numpy as np
from pydrake.all import (LinearQuadraticRegulator, DiagramBuilder, SceneGraph, WrapToSystem, LeafSystem,
                         MeshcatVisualizerCpp, Simulator, StartMeshcat, Saturation, Linearize, BasicVector)
from pydrake.examples.acrobot import (AcrobotPlant, AcrobotInput, AcrobotGeometry, AcrobotState) 
from acrobot_sos_swingup import acrobot_constrained_lqr

import matplotlib.pyplot as plt
from matplotlib import cm

meshcat = StartMeshcat()

def UprightState():
    state = AcrobotState()
    state.set_theta1(np.pi)
    state.set_theta2(0.)
    state.set_theta1dot(0.)
    state.set_theta2dot(0.)
    return state

class Controller(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        acrobot = AcrobotPlant()
        context = acrobot.CreateDefaultContext()

        input = AcrobotInput()
        input.set_tau(0.)
        acrobot.get_input_port(0).FixValue(context, input)

        context.get_mutable_continuous_state_vector()\
                    .SetFromVector(UprightState().CopyToVector())

        self.x0 = np.array([np.pi, 0, 0, 0])

        linearized_acrobot = Linearize(acrobot, context)
        self.A = linearized_acrobot.A()
        self.B = linearized_acrobot.B()
        Q = np.diag((10., 10., 1., 1.))
        R = [1]
        self.K = LinearQuadraticRegulator(self.A, self.B, Q, R)[0]
        self.state_input_port = self.DeclareVectorInputPort(
            "state", BasicVector(4))
        self.policy_output_port = self.DeclareVectorOutputPort(
            "policy", BasicVector(1), self.CalculateController)
        
    def CalculateController(self, context, output):
        x = self.state_input_port.Eval(context)
        y = output.get_mutable_value()
        y[:]  = -self.K@(x-self.x0)
        # print(y)

class ConstrainedLQR(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        self.K = acrobot_constrained_lqr()
        self.x0 = np.zeros(6)
        self.x2z = lambda x : np.array([np.sin(x[0]), np.cos(x[0]), np.sin(x[1]), np.cos(x[1]), x[2], x[3]])
        self.z0 = self.x2z(self.x0)
        self.state_input_port = self.DeclareVectorInputPort(
            "state", BasicVector(4))
        self.policy_output_port = self.DeclareVectorOutputPort(
            "policy", BasicVector(1), self.CalculateController)
        
    def CalculateController(self, context, output):
        x = self.state_input_port.Eval(context)
        y = output.get_mutable_value()
        z = self.x2z(x)
        y[:]  = -self.K@(z-self.z0)

def acrobot_balancing_example():

    def BalancingLQR():
        # Design an LQR controller for stabilizing the Acrobot around the upright.
        # Returns a (static) AffineSystem that implements the controller (in
        # the original AcrobotState coordinates).

        acrobot = AcrobotPlant()
        context = acrobot.CreateDefaultContext()

        input = AcrobotInput()
        input.set_tau(0.)
        acrobot.get_input_port(0).FixValue(context, input)

        context.get_mutable_continuous_state_vector()\
            .SetFromVector(UprightState().CopyToVector())

        Q = np.diag((10., 10., 1., 1.))
        R = [1]

        return LinearQuadraticRegulator(acrobot, context, Q, R)


    builder = DiagramBuilder()
    acrobot = builder.AddSystem(AcrobotPlant())

    saturation = builder.AddSystem(Saturation(min_value=[-10], max_value=[10]))
    builder.Connect(saturation.get_output_port(0), acrobot.get_input_port(0))
    wrapangles = WrapToSystem(4)
    wrapangles.set_interval(0, 0, 2. * np.pi)
    wrapangles.set_interval(1, -np.pi, np.pi)
    wrapto = builder.AddSystem(wrapangles)
    builder.Connect(acrobot.get_output_port(0), wrapto.get_input_port(0))
    # controller = builder.AddSystem(BalancingLQR())
    controller = builder.AddSystem(ConstrainedLQR())
    builder.Connect(wrapto.get_output_port(0), controller.get_input_port(0))
    builder.Connect(controller.get_output_port(0), saturation.get_input_port(0))

    # Setup visualization
    scene_graph = builder.AddSystem(SceneGraph())
    AcrobotGeometry.AddToBuilder(builder, acrobot.get_output_port(0), scene_graph)
    meshcat.Delete()
    meshcat.Set2dRenderMode(xmin=-4, xmax=4, ymin=-4, ymax=4)
    viz = MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat)

    diagram = builder.Build()

    # Set up a simulator to run this diagram
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()

    # Simulate
    simulator.set_target_realtime_rate(1.0)
    duration = 5.0 
    viz.StartRecording()
    for i in range(5):
        context.SetTime(0.)
        context.SetContinuousState(UprightState().CopyToVector() +
                                0.05 * np.random.randn(4,))
        # context.SetContinuousState([np.pi-0.05, 0, 0, 0])
        simulator.Initialize()
        simulator.AdvanceTo(duration)
    viz.StopRecording()
    viz.PublishRecording()

def plot_lqr_cost_to_go(x_min, x_max):
    acrobot = AcrobotPlant()
    context = acrobot.CreateDefaultContext()

    input = AcrobotInput()
    input.set_tau(0.)
    acrobot.get_input_port(0).FixValue(context, input)

    context.get_mutable_continuous_state_vector()\
                .SetFromVector(UprightState().CopyToVector())

    x0 = np.array([np.pi, 0, 0, 0])

    linearized_acrobot = Linearize(acrobot, context)
    A = linearized_acrobot.A()
    B = linearized_acrobot.B()
    Q = np.diag((10., 10., 1., 1.))
    R = [1]
    K, S = LinearQuadraticRegulator(A, B, Q, R)

    X1, X2 = np.meshgrid(np.linspace(x_min[0], x_max[0], 51),
                    np.linspace(x_min[1], x_max[1], 51))
    X = np.vstack((X1.flatten(), X2.flatten(), np.zeros(51*51), np.zeros(51*51)))

    J = np.zeros(X.shape[1])
    U = np.zeros(X.shape[1])

    for i in range(X.shape[1]):
        x = X[:, i] - x0
        J[i] = (x.dot(S)).dot(x)
        U[i] = -K.dot(x)

    fig = plt.figure(figsize=(9, 4))
    ax = fig.subplots()
    ax.set_xlabel("q1")
    ax.set_ylabel("q2")
    ax.set_title("Cost-to-Go")
    im = ax.imshow(J.reshape(X1.shape),
            cmap=cm.jet, aspect='auto',
            extent=(x_min[0], x_max[0], x_max[1], x_min[1]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.savefig("acrobot/figures/lqr_cost_to_go.png")

    fig = plt.figure(figsize=(9, 4))
    ax = fig.subplots()
    ax.set_xlabel("q1")
    ax.set_ylabel("q2")
    ax.set_title("Policy")
    im = ax.imshow(U.reshape(X1.shape),
            cmap=cm.jet, aspect='auto',
            extent=(x_min[0], x_max[0], x_max[1], x_min[1]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.savefig("acrobot/figures/lqr_policy.png")


if __name__ == '__main__':
    acrobot_balancing_example()
    # x_max = np.array([2*np.pi, np.pi/2, 3, 3])
    # x_min = np.array([0, -np.pi/2, -3, -3])
    # plot_lqr_cost_to_go(x_min, x_max)