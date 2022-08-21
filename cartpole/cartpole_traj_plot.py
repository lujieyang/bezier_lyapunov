import numpy as np
import matplotlib.pyplot as plt
from cartpole_swingup_simulate import simulate, set_orthographic_camera_xy
from cartpole_swingup_fvi import cartpole_setup
from utils import load_polynomial
from pydrake.all import (MathematicalProgram, DiagramBuilder, StartMeshcat, FindResourceOrThrow,
                         MeshcatVisualizerCpp, Simulator, AddMultibodyPlantSceneGraph, Parser)
from pydrake.geometry import Rgba
from underactuated import FindResource
from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
import meshcat
from pydrake.systems.meshcat_visualizer import (ConnectMeshcatVisualizer)

prog = MathematicalProgram()
z = prog.NewIndeterminates(5, "z")
Q_diag = [200, 2e3, 2e3, 1e3, 1e3]
z_max = [2.0, 1.0, 0.8090169943749476, 5.0, 5.0]
J_star = load_polynomial(z, "cartpole/data/{}/{}/J_iterative_0_upper_bound_lower_deg_1_deg_6.pkl".format(Q_diag, z_max))

params_dict = cartpole_setup()

x2z = lambda x : np.array([x[0], np.sin(x[1]), np.cos(x[1]), x[2], x[3]])

x_max = np.array([0.5, 1.2*np.pi, 5, 5])
x_min = np.array([-0.5, 0.8*np.pi, -5, -5])
X1, X2 = np.meshgrid(np.linspace(x_min[0], x_max[0], 3),
                    np.linspace(x_min[1], x_max[1], 6))
X = np.vstack((X1.reshape(1, 18), X2.reshape(1, 18), np.zeros([2, 18])))
t = np.arange(121)*0.05
for i in range(18):
    print(i)
    x0 = X[:, i]
    assert (x0<=x_max).all() and (x0>=x_min).all()
    x = simulate(J_star, z, params_dict, x0)
    J = []
    for j in range(x.shape[1]):
        z_val = x2z(x[:, j])
        J.append(J_star.Evaluate(dict(zip(z, z_val)))/100)
    if np.abs(J[-1]) <=1:
        print(i)
        plt.plot(t, J)
plt.xticks([0, 3 ,6], ["0", "3", "6"], fontsize=12)
plt.yticks([0, 100, 200], ["0", "100", "200"], fontsize=12)
plt.savefig("cartpole/figures/paper/J_along_trj.png")


# x0 = [0, 0.1, 0, 0]
# x = simulate(J_star, z, params_dict, x0)

# builder = DiagramBuilder()
# plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.01)
# parser = Parser(plant)

# file_name = FindResource("models/cartpole.urdf")
# cartpole1_id = parser.AddModelFromFile(file_name, "cartpole1")
# cartpole2_id = parser.AddModelFromFile(file_name, "cartpole2")
# cartpole3_id = parser.AddModelFromFile(file_name, "cartpole3")
# cartpole4_id = parser.AddModelFromFile(file_name, "cartpole4")
# cartpole5_id = parser.AddModelFromFile(file_name, "cartpole5")
# cartpole6_id = parser.AddModelFromFile(file_name, "cartpole6")
# cartpole7_id = parser.AddModelFromFile(file_name, "cartpole7")
# cartpole8_id = parser.AddModelFromFile(file_name, "cartpole8")
# cartpole9_id = parser.AddModelFromFile(file_name, "cartpole9")
# cartpole10_id = parser.AddModelFromFile(file_name, "cartpole10")
# plant.Finalize()

# # meshcat.Delete()
# # meshcat.ResetRenderMode()
# # meshcat.SetProperty('/Background','visible',False)
# # viz = MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat)
# proc, zmq_url, web_url = start_zmq_server_as_subprocess(server_args=[])
# viz = ConnectMeshcatVisualizer(builder, scene_graph, zmq_url=zmq_url)
# set_orthographic_camera_xy(viz.vis)

# diagram = builder.Build()

# # Set up a simulator to run this diagram
# simulator = Simulator(diagram)
# simulator.set_target_realtime_rate(1.0)
# context = simulator.get_mutable_context()

# context_plant = plant.GetMyMutableContextFromRoot(context)
# x1 = x[:2, 0]
# x2 = x[:2, 61]
# x3 = x[:2, 63]
# x4 = x[:2, 65]
# x5 = x[:2, 67]
# x6 = x[:2, 69]
# x7 = x[:2, 72]
# x8 = x[:2, 52]
# x9 = x[:2, 40]
# x10 = x[:2, 76]
# plant.SetPositions(context_plant, cartpole1_id, x1)
# plant.SetPositions(context_plant, cartpole2_id, x2)
# plant.SetPositions(context_plant, cartpole3_id, x3)
# plant.SetPositions(context_plant, cartpole4_id, x4)
# plant.SetPositions(context_plant, cartpole5_id, x5)
# plant.SetPositions(context_plant, cartpole6_id, x6)
# plant.SetPositions(context_plant, cartpole7_id, x7)
# plant.SetPositions(context_plant, cartpole8_id, x8)
# plant.SetPositions(context_plant, cartpole9_id, x9)
# plant.SetPositions(context_plant, cartpole10_id, x10)

# plant.get_actuation_input_port(cartpole1_id).FixValue(context_plant, [0])
# plant.get_actuation_input_port(cartpole2_id).FixValue(context_plant, [0])
# plant.get_actuation_input_port(cartpole3_id).FixValue(context_plant, [0])
# plant.get_actuation_input_port(cartpole4_id).FixValue(context_plant, [0])
# plant.get_actuation_input_port(cartpole5_id).FixValue(context_plant, [0])
# plant.get_actuation_input_port(cartpole6_id).FixValue(context_plant, [0])
# plant.get_actuation_input_port(cartpole7_id).FixValue(context_plant, [0])
# plant.get_actuation_input_port(cartpole8_id).FixValue(context_plant, [0])
# plant.get_actuation_input_port(cartpole9_id).FixValue(context_plant, [0])
# plant.get_actuation_input_port(cartpole10_id).FixValue(context_plant, [0])

# context.SetTime(0.)
# simulator.Initialize()
# simulator.AdvanceTo(0.00001)
# print()

