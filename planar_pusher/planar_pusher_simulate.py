import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils import load_polynomial
from pydrake.all import MakeVectorVariable, MathematicalProgram, SolverOptions, CommonSolverOption, Solve
from planar_pusher_sos import planar_pusher_sos_lower_bound

nz, nu, f, fx, f2x, mu_p, px, l_cost, x2z = planar_pusher_sos_lower_bound(2, test=True)
u_max = np.array([1, 1, 0.1])
u_min = np.array([0, -1, -0.1])
def simulate(dJdz, z_var, x0, T=5, dt=0.01, initial_guess=False):
    print("Simulating...")
    N = int(T/dt)
    x = x0
    traj = [x0]
    u_traj = []
    u_guess = None
    for n in range(N):
        z = x2z(x)
        dJdz_val = np.zeros(nz)
        for i in range(nz):
            dJdz_val[i] = dJdz[i].Evaluate(dict(zip(z_var, z)))
        if initial_guess and n < N - 1:
            u_guess = calc_u_initial_guess(x, N-n, dt)
        else:
            u_guess = None
        u_star = calc_optimal_conrol_nlp(z, dJdz_val, u_guess=u_guess)
        u_traj.append(u_star)
        x_dot = fx(x, u_star, float)
        x = np.copy(x) + x_dot*dt
        traj.append(x)
    return np.array(traj), np.array(u_traj)

def calc_u_initial_guess(x, H, dt):
    x_guess = np.linspace(x, np.zeros(4), H)
    x_dot = (x_guess[1] - x)/dt
    f2_val = f2x(x, float)
    u_guess =  np.linalg.lstsq(f2_val, x_dot)[0]
    for i in range(nu):
        u_guess[i] = np.clip(u_guess[i], u_min[i], u_max[i])
    return u_guess

def calc_optimal_conrol_nlp(z, dJdz_val, u_guess=None):
    prog = MathematicalProgram()
    u = prog.NewContinuousVariables(nu)
    prog.AddCost(l_cost(z, u) + dJdz_val.dot(f(z, u)))
    prog.AddConstraint(u[0] >= 0)
    prog.AddConstraint(u[-1]*u[1] >=0)
    prog.AddConstraint(mu_p**2*u[0]**2 - u[1]**2>= 0)
    prog.AddConstraint((mu_p**2*u[0]**2 - u[1]**2)*u[-1] == 0)
    # prog.AddBoundingBoxConstraint(u_min, u_max, u)
    if u_guess is not None:
        prog.SetInitialGuess(u, u_guess)

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    # assert result.is_success()

    u_star = result.GetSolution(u)

    for i in range(nu):
        u_star[i] = np.clip(u_star[i], u_min[i], u_max[i])

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
            
    for n in range(0, 10, 2):
        draw_box(traj[n], w_pusher=True)
    for n in range(10, 20, 4):
        draw_box(traj[n], w_pusher=True)
    for n in range(20, traj.shape[0], 10):
        draw_box(traj[n], w_pusher=True)
    ax.add_patch(Rectangle((-px, -px), 2*px, 2*px, 0, edgecolor='yellow', linestyle="--", linewidth=2, facecolor='none', antialiased="True"))
    plt.xlim([-0.35, 0.2])
    plt.ylim([-0.2, 0.35])
    plt.savefig("planar_pusher/figures/trajectory/trajectory_{}_mu_{}.png".format(deg, mu_p))


if __name__ == '__main__':
    deg = 2
    z_var = MakeVectorVariable(nz, "z")
    d_theta = np.pi/4
    z_max = np.array([0.15, 0.15, np.sin(d_theta), 1, px])
    J_star = load_polynomial(z_var, "planar_pusher/data/J_lower_deg_{}_mup_{}.pkl".format(deg, mu_p))
    dJdz = J_star.Jacobian(z_var)

    x0 = np.array([-0.28, 0.28, 0, 0])
    traj, u_traj = simulate(dJdz, z_var, x0, T=10,  initial_guess=True)
    plot_traj(traj, deg)
