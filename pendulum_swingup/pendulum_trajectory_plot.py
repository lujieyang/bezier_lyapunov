import numpy as np

from utils import load_polynomial
from sos_swingup import pendulum_sos_lower_bound, pendulum_lower_bound_roa
from polynomial_integration_fvi import pendulum_setup
from pydrake.examples.pendulum import (PendulumPlant)
from pydrake.all import (DiagramBuilder, Simulator, WrapToSystem, LeafSystem, LogVectorOutput,
                         BasicVector, MathematicalProgram)
from pendulum_figure import plot_roa

import matplotlib.pyplot as plt

class Controller(LeafSystem):
    def __init__(self, u_star, z, plant, params_dict):
        LeafSystem.__init__(self)
        self.plant = plant
        self.context = plant.CreateDefaultContext()
        self.x_dim = 2
        self.u_dim = 1
        self.x2z = params_dict["x2z"]
        self.u_star = u_star
        self.z = z

        self.state_input_port = self.DeclareVectorInputPort(
            "state", BasicVector(self.x_dim))

        self.policy_output_port = self.DeclareVectorOutputPort(
            "policy", BasicVector(self.u_dim), self.CalculateController)

    def CalculateController(self, context, output):
        state = self.state_input_port.Eval(context)
        state[0] = state[0] + np.pi
        z_value = self.x2z(state)
        y = output.get_mutable_value()
        y[:]  = self.u_star[0].Evaluate({z[0]: z_value[0], z[1]: z_value[1], z[2]: z_value[2]})

def simulate(u_star, z, params_dict, x0):
    # Animate the resulting policy.
    builder = DiagramBuilder()
    pendulum = builder.AddSystem(PendulumPlant())

    wrap = builder.AddSystem(WrapToSystem(2))
    wrap.set_interval(0, 0, 2*np.pi)
    builder.Connect(pendulum.get_output_port(0), wrap.get_input_port(0))
    vi_policy = Controller(u_star, z, pendulum, params_dict)
    builder.AddSystem(vi_policy)
    builder.Connect(wrap.get_output_port(0), vi_policy.get_input_port(0))
    builder.Connect(vi_policy.get_output_port(0),
                    pendulum.get_input_port(0))

    dt = 0.05
    state_logger = LogVectorOutput(wrap.get_output_port(), builder, dt)
    # state_logger = LogVectorOutput(pendulum.get_state_output_port(), builder, dt)

    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    simulator.get_mutable_context().SetContinuousState(x0)

    simulator.AdvanceTo(8)
    
    x = state_logger.FindLog(context).data()

    return x

params_dict = pendulum_setup()
nz, f, f2, Rinv, z0, l = pendulum_sos_lower_bound(2, test=True)
prog = MathematicalProgram()
z = prog.NewIndeterminates(3, "z")
J_upper = load_polynomial(z, "pendulum_swingup/data/J_upper_deg_2.pkl")
J_lower = load_polynomial(z, "pendulum_swingup/data/J_lower_deg_2.pkl")
J_star = J_upper
dJdz = J_star.Jacobian(z)
u_star = - .5 * Rinv.dot(f2.T).dot(dJdz.T)

x_max = np.array([np.pi, 5*np.pi])
x_min = - x_max

X1, X2 = np.meshgrid(np.linspace(x_min[0], x_max[0], 9),
                    np.linspace(x_min[1], x_max[1], 4))
X = np.vstack((X1.flatten(), X2.flatten()))

traj = []
for i in range(X.shape[1]):
    print(i)
    x0 = X[:, i]
    x0[0] = x0[0] + np.pi
    x = simulate(u_star, z, params_dict, x0)
    traj.append(x)
    x = np.copy(x)
    if x[0,0] == 0 and x[1,0] <0:
        x[0,0] = 2*np.pi
    plt.plot(x[0,], x[1,])

plot_roa(J_star, z)
# plt.xlabel(r"$\theta$ (rad)", fontsize=15)
# plt.ylabel(r"$\dot \theta$ (rad/s)", fontsize=15)
plt.xticks([0, 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi], ["0", r"$0.5\pi$", r"$\pi$", r"$1.5\pi$", r"$2\pi$"], fontsize=15)
plt.yticks([-5*np.pi, -2.5*np.pi, 0, 2.5*np.pi, 5*np.pi], [r"$-5\pi$", r"$-2.5\pi$", "0", r"$2.5\pi$", r"$5\pi$"], fontsize=15)
# Draw Bounding Box
horizontal_x = np.linspace(0, 2*np.pi, 100)
vertical_y = np.linspace(-2*np.pi, 2*np.pi, 100)
plt.plot(horizontal_x, -2*np.pi*np.ones(100), "k--")
plt.plot(horizontal_x, 2*np.pi*np.ones(100), "k--")
plt.plot(np.zeros(100), vertical_y, "k--")
plt.plot(2*np.pi*np.ones(100), vertical_y, "k--")
plt.savefig("pendulum_swingup/figures/traj.png")
