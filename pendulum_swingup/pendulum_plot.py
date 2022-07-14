import numpy as np
from utils import load_polynomial
from pydrake.all import MathematicalProgram

from matplotlib import cm
import matplotlib.pyplot as plt

x_max = np.array([np.pi, 2*np.pi])
x_min = - x_max

x2z = lambda x : np.array([np.sin(x[0]), np.cos(x[0]), x[1]])

X1, X2 = np.meshgrid(np.linspace(x_min[0], x_max[0], 51),
                np.linspace(x_min[1], x_max[1], 51))
X = np.vstack((X1.flatten(), X2.flatten()))
Z = x2z(X)

prog = MathematicalProgram()
z = prog.NewIndeterminates(3, "z")
J_upper = load_polynomial(z, "pendulum_swingup/data/J_upper_deg_2.pkl")
J_upper_4 = load_polynomial(z, "pendulum_swingup/data/J_upper_4_lower_deg_2.pkl")
J_upper_6 = load_polynomial(z, "pendulum_swingup/data/J_upper_6_lower_deg_2.pkl")
J_lower = load_polynomial(z, "pendulum_swingup/data/J_lower_l-a_deg_2.pkl")

def plot_roa(J_star, z, rho=[33], colors=["red"], label="deg. 2"):
    # rho = 33
    # rho_unbounded = 84.80796095199308  # J_upper
    X1, X2 = np.meshgrid(np.linspace(x_min[0], x_max[0], 51),
                    np.linspace(-2*np.pi, 2*np.pi, 51))
    X = np.vstack((X1.flatten(), X2.flatten()))
    x2z = lambda x : np.array([np.sin(x[0]), np.cos(x[0]), x[1]])
    Z = x2z(X)
    J = np.zeros(Z.shape[1])
    for i in range(Z.shape[1]):
        J[i] = J_star.Evaluate({z[0]: Z[0, i], z[1]: Z[1, i], z[2]: Z[2, i]})
    CS = plt.contour(X1 + np.pi, X2, J.reshape(X1.shape), levels=rho, colors=colors)
    # CS.collections[0].set_label(label)
    # CS.collections[0].set_label(r"$\bar J$ ROA")
    # CS.collections[1].set_label("CLF ROA")
    plt.rcParams['font.size'] = '15'
    # plt.legend()

def plot_3d():
    from mpl_toolkits.mplot3d import axes3d
    J0 = np.zeros(Z.shape[1])
    J1 = np.zeros(Z.shape[1])
    for i in range(Z.shape[1]):
        J0[i] = J_upper.Evaluate({z[0]: Z[0, i], z[1]: Z[1, i], z[2]: Z[2, i]})
        J1[i] = J_lower.Evaluate({z[0]: Z[0, i], z[1]: Z[1, i], z[2]: Z[2, i]})

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r"$\theta$", fontsize=15)
    ax.set_ylabel(r"$\dot \theta$", fontsize=15)
    c1 = ax.plot_surface(X1 + np.pi, X2, J0.reshape(X1.shape), rstride=5, cstride=5, alpha=0.8, cmap="RdBu", label=r"$\bar J$")
    c2 = ax.plot_surface(X1 + np.pi, X2, J1.reshape(X1.shape), rstride=5, cstride=5, alpha=0.8, cmap="coolwarm", label=r"$underline{J}$")
    plt.xticks([0, 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi], ["0", r"$0.5\pi$", r"$\pi$", r"$1.5\pi$", r"$2\pi$"], fontsize=15)
    plt.yticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi], [r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$"], fontsize=15)

    plt.savefig("pendulum_swingup/figures/paper/3d.png")

def plot_a_over_l():
    a = load_polynomial(z, "pendulum_swingup/data/a.pkl")
    l_val = load_polynomial(z, "pendulum_swingup/data/l_val.pkl")
    al = np.zeros(Z.shape[1])
    for i in range(Z.shape[1]):
        al[i] = a.Evaluate(dict(zip(z, Z[:, i])))/l_val.Evaluate(dict(zip(z, Z[:, i])))
    
    al[al>=1] = 0
    
    fig = plt.figure()
    ax = fig.subplots()
    # ax.set_xlabel(r"$\theta$")
    # ax.set_ylabel(r"$\dot \theta$")
    im = ax.imshow(al.reshape(X1.shape),
            cmap="seismic", aspect='auto',
            extent=(x_min[0] + np.pi, x_max[0]+ np.pi, x_max[1], x_min[1]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.xticks([0, 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi], ["0", r"$0.5\pi$", r"$\pi$", r"$1.5\pi$", r"$2\pi$"], fontsize=15)
    plt.yticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi], [r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$"], fontsize=15)
    plt.savefig("pendulum_swingup/figures/a_over_l.png")

def plot_upper_lower():
    J = np.zeros(Z.shape[1])
    for i in range(Z.shape[1]):
        z_val = Z[:, i]
        lower_value = J_lower.Evaluate(dict(zip(z, z_val)))
        # J[i] = J_upper.Evaluate(dict(zip(z, Z[:, i]))) - J_lower.Evaluate(dict(zip(z, Z[:, i])))
        J[i] = 1 - lower_value/ J_upper.Evaluate(dict(zip(z, z_val))) 
        if lower_value <= 1e-4:
            J[i] = 0

    fig = plt.figure()
    ax = fig.subplots()
    # ax.set_xlabel(r"$\theta$")
    # ax.set_ylabel(r"$\dot \theta$")
    im = ax.imshow(J.reshape(X1.shape),
            cmap="RdBu", aspect='auto',
            extent=(x_min[0] + np.pi, x_max[0]+ np.pi, x_max[1], x_min[1]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.xticks([0, 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi], ["0", r"$0.5\pi$", r"$\pi$", r"$1.5\pi$", r"$2\pi$"], fontsize=15)
    plt.yticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi], [r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$"], fontsize=15)
    # plt.savefig("pendulum_swingup/figures/paper/J_upper_over_J_lower.png")

plot_upper_lower()
# plot_a_over_l()
plot_roa(J_upper, z, rho=[33])
# plot_roa(J_upper, z, rho=[33, 84.80796095199308])
# plot_roa(J_upper_4, z, rho=[32, 77.69762180968776], colors="blueviolet", label="deg. 4")
# plot_roa(J_upper_6, z, rho=[32.5, 80.86451225961355], colors="green", label="deg. 6")
# plt.savefig("pendulum_swingup/figures/paper/roa.png")
plt.savefig("pendulum_swingup/figures/paper/J_lower_over_J_upper.png")
# plt.savefig("pendulum_swingup/figures/paper/a_over_l_roa.png")
# plot_3d()


