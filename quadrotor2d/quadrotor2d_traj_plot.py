import numpy as np
import matplotlib.pyplot as plt
from utils import calc_u_opt, load_polynomial
from quadrotor2d_sos import quadrotor2d_sos_lower_bound
from pydrake.all import (DiagramBuilder, Simulator, LogVectorOutput, LeafSystem,
                         BasicVector, MathematicalProgram)
from underactuated.quadrotor2d import Quadrotor2D

class Quadrotor2DVisualizer:
    """
    Copied from
    https://github.com/RussTedrake/underactuated/blob/master/underactuated/quadrotor2d.py
    """
    def __init__(self, ax, x_lim, y_lim):
        self.ax = ax
        self.ax.set_aspect("equal")
        self.ax.set_xlim(x_lim[0], x_lim[1])
        self.ax.set_ylim(y_lim[0], y_lim[1])

        self.length = .25  # moment arm (meters)

        self.base = np.vstack((1.2 * self.length * np.array([1, -1, -1, 1, 1]),
                               0.025 * np.array([1, 1, -1, -1, 1])))
        self.pin = np.vstack((0.005 * np.array([1, 1, -1, -1, 1]),
                              .1 * np.array([1, 0, 0, 1, 1])))
        a = np.linspace(0, 2 * np.pi, 50)
        self.prop = np.vstack(
            (self.length / 1.5 * np.cos(a), .1 + .02 * np.sin(2 * a)))

        # yapf: disable
        self.base_fill = self.ax.fill(
            self.base[0, :], self.base[1, :], zorder=1, edgecolor="k",
            facecolor=[.6, .6, .6])
        self.left_pin_fill = self.ax.fill(
            self.pin[0, :], self.pin[1, :], zorder=0, edgecolor="k",
            facecolor=[0, 0, 0])
        self.right_pin_fill = self.ax.fill(
            self.pin[0, :], self.pin[1, :], zorder=0, edgecolor="k",
            facecolor=[0, 0, 0])
        self.left_prop_fill = self.ax.fill(
            self.prop[0, :], self.prop[0, :], zorder=0, edgecolor="k",
            facecolor=[0, 0, 1])
        self.right_prop_fill = self.ax.fill(
            self.prop[0, :], self.prop[0, :], zorder=0, edgecolor="k",
            facecolor=[0, 0, 1])
        # yapf: enable

    def draw(self, x):
        R = np.array([[np.cos(x[2]), -np.sin(x[2])],
                      [np.sin(x[2]), np.cos(x[2])]])

        p = np.dot(R, self.base)
        self.base_fill[0].get_path().vertices[:, 0] = x[0] + p[0, :]
        self.base_fill[0].get_path().vertices[:, 1] = x[1] + p[1, :]

        p = np.dot(R, np.vstack(
            (-self.length + self.pin[0, :], self.pin[1, :])))
        self.left_pin_fill[0].get_path().vertices[:, 0] = x[0] + p[0, :]
        self.left_pin_fill[0].get_path().vertices[:, 1] = x[1] + p[1, :]
        p = np.dot(R, np.vstack(
            (self.length + self.pin[0, :], self.pin[1, :])))
        self.right_pin_fill[0].get_path().vertices[:, 0] = x[0] + p[0, :]
        self.right_pin_fill[0].get_path().vertices[:, 1] = x[1] + p[1, :]

        p = np.dot(
            R, np.vstack((-self.length + self.prop[0, :], self.prop[1, :])))
        self.left_prop_fill[0].get_path().vertices[:, 0] = x[0] + p[0, :]
        self.left_prop_fill[0].get_path().vertices[:, 1] = x[1] + p[1, :]

        p = np.dot(R,
                   np.vstack((self.length + self.prop[0, :], self.prop[1, :])))
        self.right_prop_fill[0].get_path().vertices[:, 0] = x[0] + p[0, :]
        self.right_prop_fill[0].get_path().vertices[:, 1] = x[1] + p[1, :]

        plt.show()

        # self.ax.set_xlabel('x (m)', fontsize=15)
        # self.ax.set_ylabel('z (m)', fontsize=15)

class Controller(LeafSystem):
    def __init__(self, J_star, z):
        LeafSystem.__init__(self)
        self.nz,  _, self.f2, self.x2z, self.Rinv, self.z0, self.u0 = quadrotor2d_sos_lower_bound(2, test=True)
        self.u_max = 2.5 * self.u0[0]
        self.u_min = 0
        self.x_dim = 6
        self.u_dim = 2
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
            dJdz_val[n] = self.dJdz[n].Evaluate(dict(zip(self.z, z_val)))
        u_opt = calc_u_opt(dJdz_val, f2_val, self.Rinv)
        y[:]  = np.clip(u_opt + self.u0, self.u_min, self.u_max )# CAUTION: Add u_equilibrium
    
def simulate(J_star, z, x0=[-1, -1, np.pi/2, 1, 1, 1]):
    # Animate the resulting policy.
    builder = DiagramBuilder()
    plant = builder.AddSystem(Quadrotor2D())
    controller = builder.AddSystem(Controller(J_star, z))
    builder.Connect(controller.get_output_port(0), plant.get_input_port(0))
    builder.Connect(plant.get_output_port(0), controller.get_input_port(0))

    dt = 0.05
    state_logger = LogVectorOutput(plant.get_output_port(), builder, dt)

    # Setup visualization
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    context.SetTime(0.)
    context.SetContinuousState(x0)
    simulator.Initialize()
    print("Simulating...")
    simulator.AdvanceTo(6)

    x = state_logger.FindLog(context).data()

    return x

prog = MathematicalProgram()
z = prog.NewIndeterminates(7, "z")
J_star = load_polynomial(z, "quadrotor2d/data/J_upper_bound_deg_2.pkl")

x_max = np.array([0.75, 0.75, np.pi/2, 1, 1, 1])
x_min = -x_max
x2z = lambda x : np.array([x[0], x[1], np.sin(x[2]), np.cos(x[2]), x[3], x[4], x[5]])

X1, X2, X3 = np.meshgrid(np.linspace(x_min[0], x_max[0], 2),
                    np.linspace(x_min[1], x_max[1], 2),
                    np.linspace(x_min[2], x_max[2], 3),)
X = np.vstack((X1.reshape(1, 12), X2.reshape(1, 12), X3.reshape(1, 12), np.ones([3, 12])))
t = np.arange(121)*0.05
for i in range(12):
    print(i)
    x0 = X[:, i]
    x = simulate(J_star, z, x0)
    z_val = x2z(x)
    J = []
    for j in range(x.shape[1]):
        J.append(J_star.Evaluate(dict(zip(z, z_val[:, j]))))
    plt.plot(t, J)

plt.xticks([0, 2 ,4, 6], ["0", "2", "4", "6"], fontsize=12)
plt.yticks([0, 20, 40, 60, 80], ["0", "20", "40", "60", "80"], fontsize=12)
plt.savefig("quadrotor2d/figures/paper/J_along_trj.png")
# fig = plt.figure()
# ax = fig.subplots()
# visualizer = Quadrotor2DVisualizer(ax, [-1.5, 1.5], [-1.5, 0.5])
# ax.plot(x[0], x[1], "r", linewidth=2)
# visualizer.draw(x[:, 0])
# visualizer1 = Quadrotor2DVisualizer(ax, [-1.5, 1.5], [-1.5, 0.5])
# visualizer1.draw(x[:, 10])
# visualizer2 = Quadrotor2DVisualizer(ax, [-1.5, 1.5], [-1.5, 0.5])
# visualizer2.draw(x[:, 20])
# visualizer3 = Quadrotor2DVisualizer(ax, [-1.5, 1.5], [-1.5, 0.5])
# visualizer3.draw(x[:, -1])
# plt.savefig("quadrotor2d/figures/paper/snapshots2.png")