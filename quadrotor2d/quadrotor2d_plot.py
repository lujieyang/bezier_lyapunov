import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from utils import load_polynomial
from pydrake.all import (MathematicalProgram)

prog = MathematicalProgram()
z = prog.NewIndeterminates(7, "z")
J_upper = load_polynomial(z, "quadrotor2d/data/J_upper_bound_deg_2.pkl")
J_lower = load_polynomial(z, "quadrotor2d/data/[1. 1. 1. 1. 1. 1. 1.]/J_lower_bound_deg_2.pkl")


def plot_upper_lower(J_upper, J_lower, z, plot_states="xy"):
    x_max = np.array([1, 1, np.pi/2, 1, 1, 1])
    x_min = -x_max

    x2z = lambda x : np.array([x[0], x[1], np.sin(x[2]), np.cos(x[2]), x[3], x[4], x[5]])

    zero_vector = np.zeros(51*51)
    if plot_states == "xtheta":
        X1, THETA = np.meshgrid(np.linspace(x_min[0], x_max[0], 51),
                        np.linspace(x_min[2], x_max[2], 51))
        X = np.vstack((X1.flatten(), zero_vector, THETA.flatten(), zero_vector, zero_vector, zero_vector))
        ylabel="theta"
        y_ind = 2
    elif plot_states == "xy":
        X1, Y = np.meshgrid(np.linspace(x_min[0], x_max[0], 51),
                        np.linspace(x_min[1], x_max[1], 51))
        X = np.vstack((X1.flatten(), Y.flatten(), zero_vector, zero_vector, zero_vector, zero_vector))
        ylabel="y"
        y_ind = 1

    Z = x2z(X)
    J = np.zeros(Z.shape[1])
    for i in range(Z.shape[1]):
        z_val = Z[:, i]
        # J[i] = J_upper.Evaluate(dict(zip(z, z_val))) - J_lower.Evaluate(dict(zip(z, z_val)))
        lower_value = J_lower.Evaluate(dict(zip(z, z_val)))
        if lower_value == 0:
            J[i] = 0
        else:
            J[i] = 1 - J_lower.Evaluate(dict(zip(z, z_val)))/ J_upper.Evaluate(dict(zip(z, z_val))) 

    fig = plt.figure()
    ax = fig.subplots()
    # ax.set_xlabel("x")
    # ax.set_ylabel(ylabel)
    # ax.set_title("Cost-to-Go")
    im = ax.imshow(J.reshape(X1.shape),
            cmap="RdBu", aspect='auto',
            extent=(x_min[0], x_max[0], x_max[y_ind], x_min[y_ind]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.xticks([-1, -0.5, 0, 0.5, 1], ["-1", "-0.5", "0", "0.5", "1"], fontsize=12)
    plt.yticks([-1, -0.5, 0, 0.5, 1], ["-1", "-0.5", "0", "0.5", "1"], fontsize=12)
    plt.savefig("quadrotor2d/figures/paper/J_lower_over_J_upper{}.png".format(plot_states))

plot_upper_lower(J_upper, J_lower, z, plot_states="xy")
