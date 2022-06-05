import numpy as np
from pydrake.all import (LinearQuadraticRegulator, DiagramBuilder, SceneGraph, WrapToSystem, LeafSystem,
                         MeshcatVisualizerCpp, Simulator, StartMeshcat, Saturation, Linearize, BasicVector)
from underactuated.quadrotor2d import Quadrotor2D, Quadrotor2DVisualizer

import matplotlib.pyplot as plt
from matplotlib import cm

meshcat = StartMeshcat()

class Controller(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        quadrotor = Quadrotor2D()
        context = quadrotor.CreateDefaultContext()

        u = quadrotor.mass * quadrotor.gravity / 2. * np.array([1, 1])
        quadrotor.get_input_port(0).FixValue(context, u)
        r = quadrotor.length

        self.x0 = np.zeros(6)
        context.get_mutable_continuous_state_vector()\
                    .SetFromVector(self.x0)

        linearized_quadrotor = Linearize(quadrotor, context)
        self.A = linearized_quadrotor.A()
        self.B = linearized_quadrotor.B()
        Q = np.diag([10, 10, 10, 1, 1, r/(2*np.pi)])
        R = np.array([[0.1, 0.05], [0.05, 0.1]])
        self.K = LinearQuadraticRegulator(self.A, self.B, Q, R)[0]
        self.state_input_port = self.DeclareVectorInputPort(
            "state", BasicVector(6))
        self.policy_output_port = self.DeclareVectorOutputPort(
            "policy", BasicVector(2), self.CalculateController)
        
    def CalculateController(self, context, output):
        x = self.state_input_port.Eval(context)
        y = output.get_mutable_value()
        y[:]  = -self.K@(x-self.x0)
        print(y)

def quadrotor_balancing_example():
    builder = DiagramBuilder()
    quadrotor = builder.AddSystem(Quadrotor2D())

    # saturation = builder.AddSystem(Saturation(min_value=[-10], max_value=[10]))
    # builder.Connect(saturation.get_output_port(0), quadrotor.get_input_port(0))
    wrapangles = WrapToSystem(6)
    wrapangles.set_interval(2, -np.pi, np.pi)
    wrapto = builder.AddSystem(wrapangles)
    builder.Connect(quadrotor.get_output_port(0), wrapto.get_input_port(0))
    # controller = builder.AddSystem(BalancingLQR())
    controller = builder.AddSystem(Controller())
    builder.Connect(wrapto.get_output_port(0), controller.get_input_port(0))
    builder.Connect(controller.get_output_port(0), quadrotor.get_input_port(0))

    # Setup visualization
    visualizer = builder.AddSystem(Quadrotor2DVisualizer())
    builder.Connect(quadrotor.get_output_port(0), visualizer.get_input_port(0))

    diagram = builder.Build()

    # Set up a simulator to run this diagram
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()

    # Simulate
    simulator.set_target_realtime_rate(1.0)
    duration = 4.0 
    for i in range(5):
        context.SetTime(0.)
        context.SetContinuousState(0.05 * np.random.randn(6))
        simulator.Initialize()
        simulator.AdvanceTo(duration)

def plot_lqr_cost_to_go(x_min, x_max):
    quadrotor = Quadrotor2D()
    context = quadrotor.CreateDefaultContext()

    u = quadrotor.mass * quadrotor.gravity / 2. * np.array([1, 1])
    quadrotor.get_input_port(0).FixValue(context, u)
    r = quadrotor.length
    quadrotor.get_input_port(0).FixValue(context, u)

    x0 = np.zeros(6)
    context.get_mutable_continuous_state_vector()\
                .SetFromVector(x0)

    linearized_quadrotor = Linearize(quadrotor, context)
    A = linearized_quadrotor.A()
    B = linearized_quadrotor.B()
    Q = np.diag([10, 10, 10, 1, 1, r/(2*np.pi)])
    R = np.array([[0.1, 0.05], [0.05, 0.1]])
    K, S = LinearQuadraticRegulator(A, B, Q, R)

    X1, THETA = np.meshgrid(np.linspace(x_min[0], x_max[0], 51),
                    np.linspace(x_min[2], x_max[2], 51))
    zero_vector = np.zeros(51*51)
    X = np.vstack((X1.flatten(), zero_vector, THETA.flatten(), zero_vector, zero_vector, zero_vector))

    J = np.zeros(X.shape[1])
    U = np.zeros(X.shape[1])

    for i in range(X.shape[1]):
        x = X[:, i] - x0
        J[i] = (x.dot(S)).dot(x)
        U[i] = -K.dot(x)[0]

    fig = plt.figure(figsize=(9, 4))
    ax = fig.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("theta")
    ax.set_title("Cost-to-Go")
    im = ax.imshow(J.reshape(X1.shape),
            cmap=cm.jet, aspect='auto',
            extent=(x_min[0], x_max[0], x_max[2], x_min[2]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.savefig("quadrotor2d/figures/lqr_cost_to_go.png")

    fig = plt.figure(figsize=(9, 4))
    ax = fig.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("theta")
    ax.set_title("Policy")
    im = ax.imshow(U.reshape(X1.shape),
            cmap=cm.jet, aspect='auto',
            extent=(x_min[0], x_max[0], x_max[2], x_min[2]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.savefig("quadrotor2d/figures/lqr_policy.png")


if __name__ == '__main__':
    # quadrotor_balancing_example()
    x_max = np.array([0.75, 0.75, np.pi/2, 4, 4, 2.75])
    x_min = -x_max
    plot_lqr_cost_to_go(x_min, x_max)