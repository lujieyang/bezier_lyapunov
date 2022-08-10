import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils import load_polynomial
from pydrake.all import MakeVectorVariable, MathematicalProgram, SolverOptions, CommonSolverOption, Solve
from planar_pusher_sos import planar_pusher_4_modes_contact_switching_sos_lower_bound

nz, nu, f, fx, f2x, mu_p, px, l_cost, l_teleport, x2z = planar_pusher_4_modes_contact_switching_sos_lower_bound(2, test=True)
u_max = np.array([1, 1, 0.1, 0.1])
u_min = np.array([0, -1, -0.1, -0.1])
def simulate(J_star, dJdz, z_var, x0, T=5, dt=0.01, initial_guess=False):
    print("Simulating...")
    N = int(T/dt)
    x = x0
    traj = [x0]
    u_traj = []
    u_guess = None
    for n in range(N):
        z = x2z(x)
        normal, tangential = calc_normal(x)
        dJdz_val = np.zeros(nz)
        for i in range(nz):
            dJdz_val[i] = dJdz[i].Evaluate(dict(zip(z_var, z)))
        if initial_guess and n < N - 1:
            u_guess = calc_u_initial_guess(x, normal, tangential, N-n, dt)
        else:
            u_guess = None
        u_star, teleport = calc_optimal_conrol(J_star, z, z_var, dJdz_val, normal, tangential, dt, u_guess=u_guess)
        if teleport:
            x = np.copy(x)
            x[-2:] = u_star
            u_traj.append(np.zeros(nu))
        else:
            u_traj.append(u_star)
            x_dot = fx(x, u_star, normal, tangential, float)
            x = np.copy(x) + x_dot*dt
        traj.append(x)
    return np.array(traj), np.array(u_traj)

def calc_u_initial_guess(x, n, d, H, dt):
    x0 = np.zeros(5)
    x0[-2:] = x[-2:]
    x_guess = np.linspace(x, x0, H)
    x_dot = (x_guess[1] - x)/dt
    f2_val = f2x(x, n, d, float)
    u_guess =  np.linalg.lstsq(f2_val, x_dot)[0]
    for i in range(nu):
        u_guess[i] = np.clip(u_guess[i], u_min[i], u_max[i])
    return u_guess

def calc_optimal_conrol(J, z, z_var, dJdz_val, n, d, dt, u_guess=None):
    u_star, Jdot = calc_optimal_conrol_nlp(z, dJdz_val, n, d, u_guess)
    J_pre = J.Evaluate(dict(zip(z_var, z)))
    J_continuous = J_pre + Jdot * dt
    z_post_star, J_post = calc_teleport_miqp(z, z_var, J, n)
    if J_continuous <= J_post:
        return u_star, False
    else:
        return z_post_star, True

def calc_teleport_miqp(z, z_var, J, n, eps=1e-3, M=1):
    prog = MathematicalProgram()
    z_post = prog.NewContinuousVariables(2, 'z_post')
    b = prog.NewBinaryVariables(4)  # Can be reduced to 3 given n
    J_post = J.EvaluatePartial(dict(zip(z_var[:4], z[:4])))
    J_post = J_post.Substitute(dict(zip(z_var[4:], z_post)))
    l_teleport_val = l_teleport(z[4:], z_post)
    prog.AddCost(l_teleport_val + J_post)
    prog.AddBoundingBoxConstraint(-px*np.ones(2), px*np.ones(2), z_post)
    prog.AddConstraint(n.dot(z_post) >= 0)
    prog.AddLinearConstraint(np.sum(b) == 1)
    # On left surface
    prog.AddConstraint(z_post[0]+px >= -(1-b[0]*M))
    prog.AddConstraint(z_post[0]+px <= (1-b[0]*M))
    # On right surface
    prog.AddConstraint(z_post[0]-px >= -(1-b[1]*M))
    prog.AddConstraint(z_post[0]-px <= (1-b[1]*M)) 
    # On bottom surface
    prog.AddConstraint(z_post[1]+px >= -(1-b[2]*M))
    prog.AddConstraint(z_post[1]+px <= (1-b[2]*M))   
    # On top surface
    prog.AddConstraint(z_post[1]-px >= -(1-b[3]*M))
    prog.AddConstraint(z_post[1]-px <= (1-b[3]*M))   

    # options = SolverOptions()
    # options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    # prog.SetSolverOptions(options)
    result = Solve(prog)
    assert result.is_success()

    z_post_star = result.GetSolution(z_post)
    J_post_star = result.GetSolution(J_post).Evaluate()

    return z_post_star, J_post_star


def calc_teleport_nlp(z, z_var, J, n, eps=1e-3):
    if np.array_equal(n, np.array([0, 1])):
        # Above
        z_post_guesses = [np.array([0, px]), np.array([px, 0]), np.array([-px, 0])]
    elif np.array_equal(n, np.array([0, -1])):
        # Below
        z_post_guesses = [np.array([0, -px]), np.array([px, 0]), np.array([-px, 0])]
    elif np.array_equal(n, np.array([1, 0])):
        # Right
        z_post_guesses = [np.array([px, 0]), np.array([0, px]), np.array([0, -px])]
    elif np.array_equal(n, np.array([-1, 0])):
        # Left
        z_post_guesses = [np.array([-px, 0]), np.array([0, px]), np.array([0, -px])]  
    teleport_cost_best = np.inf
    for z_post_guess in z_post_guesses:
        prog = MathematicalProgram()
        z_post = prog.NewContinuousVariables(2, 'z_post')
        J_pre = J.Evaluate(dict(zip(z_var, z)))
        J_post = J.EvaluatePartial(dict(zip(z_var[:4], z[:4])))
        J_post = J_post.Substitute(dict(zip(z_var[4:], z_post)))
        l_teleport_val = l_teleport(z[4:], z_post)
        prog.AddCost(l_teleport_val + J_post - J_pre)
        prog.AddBoundingBoxConstraint(-px*np.ones(2), px*np.ones(2), z_post)
        prog.AddConstraint(n.dot(z_post) >= eps-px)
        prog.AddConstraint((z_post[0]-px)*(z_post[0]+px)*(z_post[1]-px)*(z_post[1]+px) * 1e5==0)
        prog.SetInitialGuess(z_post, z_post_guess)

        options = SolverOptions()
        options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        prog.SetSolverOptions(options)
        result = Solve(prog)
        assert result.is_success()

        z_post_star = result.GetSolution(z_post)
        teleport_cost = result.get_optimal_cost()

        if teleport_cost < teleport_cost_best:
            teleport_cost_best = teleport_cost
            z_post_star_best = z_post_star

        return z_post_star_best, teleport_cost_best

def calc_optimal_conrol_nlp(z, dJdz_val, n, d, u_guess=None):
    prog = MathematicalProgram()
    u = prog.NewContinuousVariables(nu)
    prog.AddCost(l_cost(z, u) + dJdz_val.dot(f(z, u, n, d)))
    if n[1] == 0:
        prog.AddConstraint(u[-2] == 0)
    elif n[0] == 0:
        prog.AddConstraint(u[-1] == 0)
    # Scaling for better numerics: length << 1
    prog.AddConstraint(u[0]*1e5 >= 0)
    prog.AddConstraint(u[-1]*u[1]*1e5 >=0)
    prog.AddConstraint(u[-2]*u[1]*1e5 >=0)
    prog.AddConstraint((mu_p**2*u[0]**2 - u[1]**2)*1e5>= 0)
    prog.AddConstraint((mu_p**2*u[0]**2 - u[1]**2)*u[-1]*1e5 == 0)
    prog.AddConstraint((mu_p**2*u[0]**2 - u[1]**2)*u[-2]*1e5 == 0)
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

    # return u_star, dJdz_val.dot(f(z, u_star, n, d, float))
    return u_star, result.get_optimal_cost()

def calc_normal(x):
    xp = x[3]
    yp = x[4]
    if xp == -px:
        n = np.array([1, 0])
        d = np.array([0, 1])
    elif xp == px:
        n = np.array([-1, 0])
        d = np.array([0, -1])
    elif yp == -px:
        n = np.array([0, 1])
        d = np.array([-1, 0])
    elif yp == px:
        n = np.array([0, -1])
        d = np.array([1, 0])
    else:
        raise Exception("Invalid configuration!")
    return n, d

def plot_traj(ax, traj, pusher_color="r"):
    print("Plotting trajectory")
    plt.plot(traj[:, 0], traj[:, 1], 'k')

    def draw_box(x, w_pusher=False, r=0.01, object_color="b", pusher_color="r"):
        theta = x[2]
        x_object = x[0] - px*(np.cos(theta) - np.sin(theta))
        y_object = x[1] - px* (np.cos(theta) + np.sin(theta))
        ax.add_patch(Rectangle((x_object, y_object), 2*px, 2*px, theta/np.pi*180, edgecolor=object_color, facecolor='none', antialiased="True"))
        if w_pusher:
            py = x[-1]
            xp = x[-2]
            if py == -px:
                x_r = -r * np.sin(theta)
                y_r = -r * np.cos(theta)
            elif py == px:
                x_r = -r * np.sin(theta)
                y_r = r * np.cos(theta)     
            elif xp == px:
                x_r = r * np.cos(theta)
                y_r = r * np.sin(theta)     
            elif xp == -px:
                x_r = -r * np.cos(theta)
                y_r = -r * np.sin(theta)          
            x_pusher = x[0] + xp*np.cos(theta) - py*np.sin(theta) + x_r
            y_pusher = x[1] + xp*np.sin(theta) + py*np.cos(theta) + y_r
            drawing_colored_circle = plt.Circle((x_pusher, y_pusher), r, edgecolor='k', facecolor=pusher_color)
            ax.set_aspect( 1 )
            ax.add_artist(drawing_colored_circle)
    
    for n in range(1, 21, 10):
        draw_box(traj[n], w_pusher=True, pusher_color=pusher_color)
    for n in range(30, traj.shape[0], 35):
        draw_box(traj[n], w_pusher=True, pusher_color=pusher_color)
    ax.add_patch(Rectangle((-px, -px), 2*px, 2*px, 0, edgecolor='deeppink', linestyle="--", linewidth=2, facecolor='none', antialiased="True"))
    plt.xlim([-0.35, 0.2])
    plt.ylim([-0.2, 0.35])
    # plt.savefig("planar_pusher/figures/trajectory/four_modes/trajectory_{}.png".format(deg, mu_p))

if __name__ == '__main__':
    deg = 2
    Q_teleport_scale = 1
    z_var = MakeVectorVariable(nz, "z")
    J_star = load_polynomial(z_var, "planar_pusher/data/four_modes/{}/J_lower_deg_2_100.pkl".format(Q_teleport_scale))
    dJdz = J_star.Jacobian(z_var)

    x0 = np.array([-0.28, 0, 0, -px, 0])
    traj, u_traj = simulate(J_star, dJdz, z_var, x0, T=10,  initial_guess=True)
    traj[:, 1] += 0.28
    fig, ax = plt.subplots()
    plot_traj(ax, traj)
    x0 = np.array([0, 0.28, 0, -px, 0])
    traj, u_traj = simulate(J_star, dJdz, z_var, x0, T=10,  initial_guess=True)
    plot_traj(ax, traj, pusher_color="green")
    plt.savefig("planar_pusher/figures/trajectory/four_modes/2_stops.png".format(deg, mu_p))
