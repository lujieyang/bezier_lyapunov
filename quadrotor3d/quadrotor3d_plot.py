import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from utils import load_polynomial
from pydrake.all import (MathematicalProgram)
from scipy.spatial.transform import Rotation as R

prog = MathematicalProgram()
z = prog.NewIndeterminates(13, "z")
J_upper = load_polynomial(z, "quadrotor3d/data/J_upper_bound_deg_2.pkl")
J_lower = load_polynomial(z, "quadrotor3d/data/[0.0, 0.63, 0.04, 0.63, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]/J_lower_bound_deg_2.pkl")

def rpy_to_quaternion(X):
    assert X.shape[1] == 3
    r = R.from_euler("zyx", X)
    xyzw = r.as_quat()
    q = np.hstack((xyzw[:, -1].reshape(X.shape[0], 1)-1,xyzw[:, :3]))
    return q

def plot_value_function(J_upper, J_lower, z, plot_states="xroll"):
    x_max = np.ones(12)
    x_max[3:6] = np.array([np.pi, 0.4 * np.pi, np.pi])
    x_min = - x_max
    xyzdot_w = np.zeros([51*51, 6])
    if plot_states == "xroll":
        x_limit_idx = 0
        y_limit_idx = 3
        X, Roll = np.meshgrid(np.linspace(x_min[x_limit_idx], x_max[x_limit_idx], 51),
                        np.linspace(x_min[y_limit_idx], x_max[y_limit_idx], 51))
        Rpy = np.hstack((Roll.reshape(51*51, 1), np.zeros([51*51, 2])))
        qwxyz = rpy_to_quaternion(Rpy)
        Z = np.hstack((qwxyz, X.reshape(51*51, 1), np.zeros([51*51, 2]), xyzdot_w))
    elif plot_states == "zpitch":
        x_limit_idx = 2
        y_limit_idx = 4
        Zz, Pitch = np.meshgrid(np.linspace(x_min[x_limit_idx], x_max[x_limit_idx], 51),
                        np.linspace(x_min[y_limit_idx], x_max[y_limit_idx], 51))
        Rpy = np.hstack((np.zeros([51*51, 1]), Pitch.reshape(51*51, 1), np.zeros([51*51, 1])))
        qwxyz = rpy_to_quaternion(Rpy)
        Z = np.hstack((qwxyz, np.zeros([51*51, 2]), Zz.reshape(51*51, 1), xyzdot_w))

    J = np.zeros(Z.shape[0])
    for i in range(Z.shape[0]):
        z_val = np.squeeze(Z[i])
        lower_value = J_lower.Evaluate(dict(zip(z, z_val)))
        # J[i] = J_upper.Evaluate(dict(zip(z, Z[:, i]))) - J_lower.Evaluate(dict(zip(z, Z[:, i])))
        if lower_value <= 1e-6:
            J[i] = 0
        else:
            J[i] = 1 - lower_value/ J_upper.Evaluate(dict(zip(z, z_val))) 

    fig = plt.figure()
    ax = fig.subplots()
    # ax.set_xlabel("x")
    # ax.set_ylabel(ylabel)
    # ax.set_title("Cost-to-Go")
    im = ax.imshow(J.reshape(51, 51),
            cmap="RdBu", aspect='auto',
            extent=(x_min[4], x_max[4], x_max[y_limit_idx], x_min[y_limit_idx]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.rcParams['font.size'] = '12'
    plt.savefig("quadrotor3d/figures/paper/J_upper_over_J_lower_{}.png".format(plot_states))

plot_value_function(J_upper, J_lower, z, plot_states="zpitch")