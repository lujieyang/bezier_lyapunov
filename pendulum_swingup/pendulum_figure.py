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
J_lower = load_polynomial(z, "pendulum_swingup/data/J_lower_deg_2.pkl")

def plot_roa(J_star, z):
    rho = 33
    rho_unbounded = 84.80796095199308  # J_upper
    X1, X2 = np.meshgrid(np.linspace(x_min[0], x_max[0], 51),
                    np.linspace(-2*np.pi, 2*np.pi, 51))
    X = np.vstack((X1.flatten(), X2.flatten()))
    x2z = lambda x : np.array([np.sin(x[0]), np.cos(x[0]), x[1]])
    Z = x2z(X)
    J = np.zeros(Z.shape[1])
    for i in range(Z.shape[1]):
        J[i] = J_star.Evaluate({z[0]: Z[0, i], z[1]: Z[1, i], z[2]: Z[2, i]})
    plt.contour(X1 + np.pi, X2, J.reshape(X1.shape), levels=[rho, rho_unbounded], colors="red")

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

    plt.savefig("pendulum_swingup/figures/3d.png")

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
        J[i] = J_upper.Evaluate(dict(zip(z, Z[:, i]))) - J_lower.Evaluate(dict(zip(z, Z[:, i])))
    
    fig = plt.figure()
    ax = fig.subplots()
    # ax.set_xlabel(r"$\theta$")
    # ax.set_ylabel(r"$\dot \theta$")
    im = ax.imshow(J.reshape(X1.shape),
            cmap=cm.jet, aspect='auto',
            extent=(x_min[0] + np.pi, x_max[0]+ np.pi, x_max[1], x_min[1]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.xticks([0, 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi], ["0", r"$0.5\pi$", r"$\pi$", r"$1.5\pi$", r"$2\pi$"], fontsize=15)
    plt.yticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi], [r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$"], fontsize=15)
    plt.savefig("pendulum_swingup/figures/J_upper-J_lower.png")

plot_a_over_l()
plot_roa(J_upper, z)
plt.savefig("pendulum_swingup/figures/paper/a_over_l.png")
# plot_3d()


