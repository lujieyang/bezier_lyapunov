import numpy as np
import matplotlib.pyplot as plt
from quadrotor3d_balance_simulate import simulate, ToTrigState
from utils import load_polynomial
from pydrake.all import (MathematicalProgram, DiagramBuilder, StartMeshcat, FindResourceOrThrow,
                         MeshcatVisualizerCpp, Simulator, AddMultibodyPlantSceneGraph, Parser)
from pydrake.geometry import Rgba

prog = MathematicalProgram()
z = prog.NewIndeterminates(13, "z")
J_star = load_polynomial(z, "quadrotor3d/data/J_upper_bound_deg_2.pkl")

x_max = np.ones(12)
x_max[3:6] = np.array([np.pi/2, 0.2 * np.pi, np.pi/2])
x_min = - x_max
X1, X3, X5 = np.meshgrid(np.linspace(x_min[0], x_max[0], 2),
                    np.linspace(x_min[2], x_max[2], 2),
                    np.linspace(x_min[4], x_max[4], 3),)
X = np.vstack((X1.reshape(1, 12), np.ones([1, 12]), X3.reshape(1, 12), np.pi/2*np.ones([1, 12]), X5.reshape(1, 12), np.ones([7, 12])))
t = np.arange(101)*0.05
for i in range(12):
    print(i)
    x0 = X[:, i]
    assert (x0<=x_max).all() and (x0>=x_min).all()
    x = simulate(J_star, z, x0)
    J = []
    for j in range(x.shape[1]):
        z_val = ToTrigState(x[:, j])
        J.append(J_star.Evaluate(dict(zip(z, z_val))))
    plt.plot(t, J)
plt.xticks([0, 2.5, 5], ["0", "2.5", "5"], fontsize=12)
plt.yticks([0, 200, 400], ["0", "200", "400"], fontsize=12)
plt.savefig("quadrotor3d/figures/paper/J_along_trj.png")


# x = simulate(J_star, z)
# meshcat = StartMeshcat()

# builder = DiagramBuilder()
# plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.01)
# parser = Parser(plant)

# quadrotor1_id = parser.AddModelFromFile(FindResourceOrThrow("drake/examples/quadrotor/quadrotor.urdf"),
#       "quadrotor1")
# quadrotor2_id = parser.AddModelFromFile(FindResourceOrThrow("drake/examples/quadrotor/quadrotor.urdf"),
#       "quadrotor2")
# quadrotor3_id = parser.AddModelFromFile(FindResourceOrThrow("drake/examples/quadrotor/quadrotor.urdf"),
#       "quadrotor3")
# quadrotor4_id = parser.AddModelFromFile(FindResourceOrThrow("drake/examples/quadrotor/quadrotor.urdf"),
#       "quadrotor4")
# plant.Finalize()

# meshcat.Delete()
# meshcat.ResetRenderMode()
# meshcat.SetProperty('/Background','visible',False)
# viz = MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat)

# diagram = builder.Build()

# # Set up a simulator to run this diagram
# simulator = Simulator(diagram)
# simulator.set_target_realtime_rate(1.0)
# context = simulator.get_mutable_context()

# context_plant = plant.GetMyMutableContextFromRoot(context)
# x1 = ToTrigState(x[:, 0])[:7]
# x2 = ToTrigState(x[:, 30])[:7]
# x3 = ToTrigState(x[:, 60])[:7]
# x4 = ToTrigState(x[:, -1])[:7]
# x1[3] += 1
# x2[3] += 1
# x3[3] += 1
# x4[3] += 1
# plant.SetPositions(context_plant, quadrotor1_id, x1)
# plant.SetPositions(context_plant, quadrotor2_id, x2)
# plant.SetPositions(context_plant, quadrotor3_id, x3)
# plant.SetPositions(context_plant, quadrotor4_id, x4)

# traj = x[:3].astype(np.float32)
# meshcat.SetLine("path", traj,rgba=Rgba(1,0,0), line_width=4)
# simulator.Initialize()
# simulator.AdvanceTo(0.00001)
# print()

