import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from utils import load_polynomial
from pydrake.all import (MathematicalProgram)

prog = MathematicalProgram()
z = prog.NewIndeterminates(5, "z")
J_upper = load_polynomial(z, "cartpole/data/{}/{}/J_iterative_0_upper_bound_lower_deg_1_deg_6.pkl".format([200, 2e3, 2e3, 1e3, 1e3], [2.0, 1.0, 0.8090169943749476, 5.0, 5.0]))
J_lower = load_polynomial(z, "cartpole/data/{}/{}/J_lower_bound_deg_{}.pkl".format([200, 2e3, 2e3, 1e3, 1e3], [2, 1, 1, 6, 6], 6))
a = load_polynomial(z, "cartpole/data/sos/a.pkl")
l_val = load_polynomial(z, "cartpole/data/sos/l_val.pkl")

def plot_upper_lower(J_upper, J_lower, z):
    x_max = np.array([2, 1.8*np.pi, 5, 5])
    x_min = np.array([-2, 0.2*np.pi, -5, -5])
    x2z = lambda x : np.array([x[0], np.sin(x[1]), np.cos(x[1]), x[2], x[3]])

    X1, X2 = np.meshgrid(np.linspace(x_min[0], x_max[0], 51),
                    np.linspace(x_min[1], x_max[1], 51))
    X = np.vstack((X1.flatten(), X2.flatten(), np.zeros(51*51), np.zeros(51*51)))

    Z = x2z(X)
    J = np.zeros(Z.shape[1])
    for i in range(Z.shape[1]):
        z_val = Z[:, i]
        J[i] = J_upper.Evaluate(dict(zip(z, z_val)))
        lower_value = J_lower.Evaluate(dict(zip(z, z_val)))
        if np.abs(lower_value) <= 1e-6:
            J[i] = 0
        else:
            J[i] = 1 - lower_value/ J_upper.Evaluate(dict(zip(z, z_val))) 
    # J[J<=0] = np.abs(J[J<=0])

    fig = plt.figure()
    ax = fig.subplots()
    im = ax.imshow(J.reshape(X1.shape),
            cmap="RdBu", aspect='auto',
            extent=(x_min[0], x_max[0], x_max[1], x_min[1]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.xticks([-2, -1, 0, 1, 2], ["-2", "-1", "0", "1", "2"], fontsize=12)
    plt.yticks([0.2*np.pi, 0.6*np.pi, np.pi, 1.4*np.pi, 1.8*np.pi], [r"$0.2\pi$", r"$0.6\pi$", r"$\pi$", r"$1.4\pi$", r"$1.8\pi$"], fontsize=12)
    plt.savefig("cartpole/figures/paper/J_lower_over_J_upper.png")
    # plt.savefig("cartpole/figures/paper/a_over_l.png")
    # plt.savefig("cartpole/figures/paper/J_upper-J_lower.png")

plot_upper_lower(J_upper, J_lower, z)
