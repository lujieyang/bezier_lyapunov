# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from pydrake.all import (DiagramBuilder, DynamicProgrammingOptions, FittedValueIteration, 
                         PeriodicBoundaryCondition, SceneGraph, Simulator,WrapToSystem,
                         AddMultibodyPlantSceneGraph, ConnectMeshcatVisualizer, Parser)

from underactuated import FindResource
# %%
from meshcat.servers.zmqserver import start_zmq_server_as_subprocess

# %%
def cartpole_swingup_example():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    file_name = FindResource("models/cartpole.urdf")
    Parser(plant).AddModelFromFile(file_name)
    plant.Finalize()

    # builder.ExportInput(plant.get_applied_generalized_force_input_port(),
    #                   "extra_input")
    builder.ExportInput(plant.get_actuation_input_port(), "cartpole_input")
    diagram = builder.Build()

    simulator = Simulator(diagram)
    options = DynamicProgrammingOptions()
    n_mesh = 41
    xbins = np.linspace(-2., 2., n_mesh)
    qbins = np.linspace(0., 2. * np.pi, n_mesh)
    qdotbins = np.linspace(-3., 3., n_mesh)
    state_grid = [set(xbins), set(qbins), set(qdotbins), set(qdotbins)]
    options.periodic_boundary_conditions = [
        PeriodicBoundaryCondition(1, 0., 2. * np.pi),
    ]
    options.discount_factor = .999
    input_limit = 10.
    input_grid = [set(np.linspace(-input_limit, input_limit, 20))]
    timestep = 0.01
    
    def quadratic_regulator_cost(context):
        x = context.get_continuous_state_vector().CopyToVector()
        x[1] = x[1] - np.pi
        u = diagram.get_input_port().Eval(context)
        return 2 * x.dot(x) + u.dot(u)

    cost_function = quadratic_regulator_cost
    options.convergence_tol = 0.1
    policy, cost_to_go = FittedValueIteration(simulator, cost_function,
                                              state_grid, input_grid, timestep,
                                              options)
    J = np.reshape(cost_to_go, (n_mesh, n_mesh, n_mesh, n_mesh))
    # np.save("pendulum_swingup/data/cartpole/J_mesh_{}".format(n_mesh), J)

    # plot_surface(meshcat, 'Cost-to-go', Q1, Q2, J[:, :, 0, 0], wireframe=True)
    fig = plt.figure(figsize=(9, 4))
    ax1, ax2 = fig.subplots(1, 2)
    ax1.set_xlabel("x")
    ax1.set_ylabel("q")
    ax1.set_title("Cost-to-Go")
    ax2.set_xlabel("x")
    ax2.set_ylabel("q")
    ax2.set_title("Policy")
    im1 = ax1.imshow(J[:, :, 0, 0],
               cmap=cm.jet, aspect='auto',
               extent=(xbins[0], xbins[-1], qbins[-1], qbins[0]))
    ax1.invert_yaxis()
    fig.colorbar(im1)
    Pi = np.reshape(policy.get_output_values(), (n_mesh, n_mesh, n_mesh, n_mesh))
    im2 = ax2.imshow(Pi[:, :, 0, 0],
               cmap=cm.jet, aspect='auto',
               extent=(xbins[0], xbins[-1], qbins[-1], qbins[0]))
    ax2.invert_yaxis()
    fig.colorbar(im2)
    plt.show()
    plt.savefig("cartpole_optimal_cost_to_go.png")
    return policy, cost_to_go

# %%
def simulate(policy):
    # Animate the resulting policy.
    builder = DiagramBuilder()
    cartpole, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    file_name = FindResource("models/cartpole.urdf")
    Parser(cartpole).AddModelFromFile(file_name)
    cartpole.Finalize()

    wrap = builder.AddSystem(WrapToSystem(4))
    wrap.set_interval(1, 0, 2*np.pi)
    builder.Connect(cartpole.get_state_output_port(), wrap.get_input_port(0))
    vi_policy = builder.AddSystem(policy)
    builder.Connect(wrap.get_output_port(0), vi_policy.get_input_port(0))
    builder.Connect(vi_policy.get_output_port(0),
                    cartpole.get_actuation_input_port())

    # Setup visualization
    proc, zmq_url, web_url = start_zmq_server_as_subprocess(server_args=[])
    viz = ConnectMeshcatVisualizer(builder, scene_graph, zmq_url=zmq_url)
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.get_mutable_context().SetContinuousState([np.pi-0.01, 0.0, 0, 0])
    viz.start_recording()
    simulator.AdvanceTo(10)
    viz.publish_recording()

# %%
policy, cost_to_go = cartpole_swingup_example()

# %%
print('Simulating...')
simulate(policy)