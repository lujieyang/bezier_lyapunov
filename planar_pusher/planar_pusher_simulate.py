import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sympy import Circle
from utils import load_polynomial
from pydrake.all import MakeVectorVariable, MathematicalProgram, SolverOptions, CommonSolverOption, Solve
from planar_pusher_sos import planar_pusher_sos_lower_bound

nz, nu, f, fx, mu_p, px, l_cost, x2z = planar_pusher_sos_lower_bound(2, test=True)
u_max =np.array([1, 1, 0.1])
u_min = -u_max
def simulate(dJdz, z_var, x0, T=5, dt=0.01):
    print("Simulating...")
    N = int(T/dt)
    x = x0
    traj = [x0]
    for n in range(N):
        z = x2z(x)
        dJdz_val = np.zeros(nz)
        for i in range(nz):
            dJdz_val[i] = dJdz[i].Evaluate(dict(zip(z_var, z)))
        u_star = calc_optimal_conrol_nlp(z, dJdz_val)
        dx = fx(x, u_star, float)
        x = np.copy(x) + dx*dt
        traj.append(x)
    return np.array(traj)

def calc_optimal_conrol_nlp(z, dJdz_val):
    prog = MathematicalProgram()
    u = prog.NewContinuousVariables(nu)
    prog.AddCost(l_cost(z, u) + dJdz_val.dot(f(z, u)))
    prog.AddConstraint(u[0] >= 0)
    prog.AddConstraint(u[-1]*u[1] >=0)
    prog.AddConstraint(mu_p**2*u[0]**2 - u[1]**2>= 0)
    prog.AddConstraint((mu_p**2*u[0]**2 - u[1]**2)*u[-1] == 0)
    # prog.AddBoundingBoxConstraint(u_min, u_max, u)

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    # assert result.is_success()

    u_star = result.GetSolution(u)

    # for i in range(nu):
    #     u_star[i] = np.clip(u_star[i], u_min[i], u_max[i])

    return u_star

def plot_traj(traj, deg):
    print("Plotting trajectory")
    fig, ax = plt.subplots()
    plt.plot(traj[:, 0], traj[:, 1], 'k')

    def draw_box(x, w_pusher=False, r=0.01):
        theta = x[2]
        x_object = x[0] - px*(np.cos(theta) - np.sin(theta))
        y_object = x[1] - px* (np.cos(theta) + np.sin(theta))
        ax.add_patch(Rectangle((x_object, y_object), 2*px, 2*px, theta/np.pi*180, edgecolor='b', facecolor='none', antialiased="True"))
        if w_pusher:
            py = x[-1]
            x_pusher = x[0] - px*np.cos(theta) - py*np.sin(theta) - r * np.cos(theta)
            y_pusher = x[1] - px*np.sin(theta) + py*np.cos(theta) - r * np.sin(theta)
            drawing_colored_circle = plt.Circle((x_pusher, y_pusher), r, edgecolor='k', facecolor='r')
            ax.set_aspect( 1 )
            ax.add_artist(drawing_colored_circle)
            
    for n in range(0, traj.shape[0], 5):
        draw_box(traj[n], w_pusher=True)
    plt.xlim([-0.35, 0.2])
    plt.ylim([-0.2, 0.35])
    plt.savefig("planar_pusher/figures/trajectory/trajectory_{}.png".format(deg))


if __name__ == '__main__':
    deg = 2
    z_var = MakeVectorVariable(nz, "z")
    d_theta = np.pi/4
    z_max = np.array([0.15, 0.15, np.sin(d_theta), 1, px])
    J_star = load_polynomial(z_var, "planar_pusher/data/J_lower_deg_{}_{}.pkl".format(deg, z_max))
    dJdz = J_star.Jacobian(z_var)

    x0 = np.array([-0.2, 0.2, -np.pi/4, 0])
    traj = simulate(dJdz, z_var, x0)
    plot_traj(traj, deg)
