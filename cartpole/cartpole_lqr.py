import numpy as np
from pydrake.all import (LinearQuadraticRegulator, DiagramBuilder, AddMultibodyPlantSceneGraph,
                         MeshcatVisualizerCpp, Simulator, StartMeshcat, Parser, LeafSystem, Linearize,
                         BasicVector)
from underactuated import FindResource
from cartpole_sos_swingup import cartpole_constrained_lqr, cartpole_lqr

import matplotlib.pyplot as plt
from matplotlib import cm

meshcat = StartMeshcat()

def UprightState():
    state = (0, np.pi, 0, 0)
    return state

class Controller(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        file_name = FindResource("models/cartpole.urdf")
        Parser(plant).AddModelFromFile(file_name)
        plant.Finalize()

        context = plant.CreateDefaultContext()
        context.get_mutable_continuous_state_vector()\
                    .SetFromVector(UprightState())
        plant.get_actuation_input_port().FixValue(context, np.array([0]))

        self.x0 = np.array([0, np.pi, 0, 0])

        linearized_plant = Linearize(
            plant,
            context,
            input_port_index=plant.get_actuation_input_port().get_index(), output_port_index=plant.get_state_output_port().get_index())

        self.A = linearized_plant.A()
        self.B = linearized_plant.B()
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

class ConstrainedLQR(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        self.K = cartpole_constrained_lqr()[0]
        self.x0 = np.array([0, np.pi, 0, 0])
        self.x2z = lambda x : np.array([x[0], np.sin(x[1]), np.cos(x[1]), x[2], x[3]])
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

def cartpole_balancing_example():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    file_name = FindResource("models/cartpole.urdf")
    Parser(plant).AddModelFromFile(file_name)
    plant.Finalize()

    # controller = builder.AddSystem(BalancingLQR(plant))
    controller = builder.AddSystem(ConstrainedLQR())
    builder.Connect(plant.get_state_output_port(), controller.get_input_port(0))
    builder.Connect(controller.get_output_port(0),
                    plant.get_actuation_input_port())

    # Setup visualization
    meshcat.Delete()
    meshcat.Set2dRenderMode(xmin=-2.5, xmax=2.5, ymin=-1.0, ymax=2.5)
    viz = MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat)

    diagram = builder.Build()

    # Set up a simulator to run this diagram
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()

    # Simulate
    simulator.set_target_realtime_rate(1.0)
    duration = 5.0 
    viz.StartRecording()
    for i in range(1):
        context.SetTime(0.)
        context.SetContinuousState([0, np.pi-1, 0, 0])
        simulator.Initialize()
        simulator.AdvanceTo(duration)
    viz.StopRecording()
    viz.PublishRecording()

def plot_lqr_cost_to_go(x_min, x_max):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    file_name = FindResource("models/cartpole.urdf")
    Parser(plant).AddModelFromFile(file_name)
    plant.Finalize()

    context = plant.CreateDefaultContext()
    context.get_mutable_continuous_state_vector()\
                .SetFromVector(UprightState())
    plant.get_actuation_input_port().FixValue(context, np.array([0]))

    x0 = np.array([0, np.pi, 0, 0])

    linearized_plant = Linearize(
        plant,
        context,
        input_port_index=plant.get_actuation_input_port().get_index(), output_port_index=plant.get_state_output_port().get_index())

    A = linearized_plant.A()
    B = linearized_plant.B()
    Q = np.diag((2., 2., 1., 1.))
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
            extent=(x_min[0], x_max[0], x_min[1], x_max[1]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.savefig("cartpole/figures/lqr_cost_to_go.png")

    fig = plt.figure(figsize=(9, 4))
    ax = fig.subplots()
    ax.set_xlabel("q1")
    ax.set_ylabel("q2")
    ax.set_title("Policy")
    im = ax.imshow(U.reshape(X1.shape),
            cmap=cm.jet, aspect='auto',
            extent=(x_min[0], x_max[0], x_min[1], x_max[1]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.savefig("cartpole/figures/lqr_policy.png")

if __name__ == '__main__':
    cartpole_balancing_example()
    # x_max = np.array([2, 2*np.pi, 3, 3])
    # x_min = np.array([-2, 0, -3, -3])
    # plot_lqr_cost_to_go(x_min, x_max)