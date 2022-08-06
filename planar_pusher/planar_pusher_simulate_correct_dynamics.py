import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils import construct_monomial_basis_from_polynomial, construct_polynomial_from_coeff, save_polynomial, load_polynomial
from pydrake.all import Variables, MathematicalProgram, SolverOptions, CommonSolverOption, Solve, MakeVectorVariable, Expression
from planar_pusher_sos import planar_pusher_sos_lower_bound

nz, nx, nu, f, fx, f2x, mu_p, px, l_cost, x2z = planar_pusher_sos_lower_bound(2, test=True)
u_max = np.array([1, 1, 0.1])
u_min = np.array([0, -1, -0.1])
def simulate(J_star, z_var, x0, T=5, dt=0.01, initial_guess=False):
    print("Simulating...")
    N = int(T/dt)
    x = x0
    traj = [x0]
    u_traj = []
    J_traj = []
    u_guess = None
    dJdz = J_star.Jacobian(z_var)
    for n in range(N):
        z = x2z(x)
        dJdz_val = np.zeros(nz)
        for i in range(nz):
            dJdz_val[i] = dJdz[i].Evaluate(dict(zip(z_var, z)))
        if initial_guess and n < N - 1:
            # u_guess = calc_u_initial_guess(x, N-n, dt)
            u_guess = calc_u_initial_guess(x, n, dt)
        else:
            u_guess = None
        u_star = calc_optimal_conrol_nlp(z, dJdz_val, u_guess=u_guess)
        u_traj.append(u_star)
        x_dot = fx(x, u_star, float)
        x = np.copy(x) + x_dot*dt
        traj.append(x)
        J_traj.append(J_star.Evaluate(dict(zip(z_var, z))))
    return np.array(traj), np.array(u_traj), np.array(J_traj)

def calc_u_initial_guess(x, H, dt):
    # x_guess = np.linspace(x, np.zeros(4), H)
    x_guess = traj_guess[H:]
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
    
    # print(result.get_optimal_cost())

    return u_star

def traj_opt(x0, T=5, dt=0.01):
    N = int(T/dt)
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(4, N+1)
    u = prog.NewContinuousVariables(nu, N)
    prog.AddBoundingBoxConstraint(x0, x0, x[:, 0])
    for n in range(N):
        x_next = x[:, n] + fx(x[:, n], u[:, n])*dt
        for i in range(nx):
            prog.AddConstraint(x_next[i] == x[i, n+1])
        prog.AddConstraint(u[0, n] >= 0)
        prog.AddConstraint(u[-1, n]*u[1, n] >=0)
        prog.AddConstraint(mu_p**2*u[0, n]**2 - u[1, n]**2>= 0)
        prog.AddConstraint((mu_p**2*u[0, n]**2 - u[1, n]**2)*u[-1, n] == 0)
        prog.AddBoundingBoxConstraint(u_min, u_max, u[:, n])
        prog.AddConstraint(x[-1, n] >= -px)
        prog.AddConstraint(x[-1, n] <= px)
    prog.AddBoundingBoxConstraint(np.zeros(3), np.zeros(3), x[:-1, -1])
    prog.AddBoundingBoxConstraint(-px, px, x[-1, -1])

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    # assert result.is_success()

    x_star = result.GetSolution(x)
    u_star = result.GetSolution(u)    
    return x_star, u_star, result.is_success()

def simulate_traj_opt(u, z_var, x0, T=5, dt=0.01):
    print("Simulating...")
    N = int(T/dt)
    x = x0
    traj = [x0]
    u_traj = []
    for n in range(N):
        z = x2z(x)
        u_star = np.zeros(nu)
        for i in range(nu):
            u_star[i] = u[i].Evaluate(dict(zip(z_var, z)))
        u_traj.append(u_star)
        x_dot = fx(x, u_star, float)
        x = np.copy(x) + x_dot*dt
        traj.append(x)
    return np.array(traj), np.array(u_traj)

def fit_polynomial_stabilizing_controller(deg1, deg2, deg3, collect_data=False):
    if collect_data:
        n_grid = 10
        X2, X3, X4 = np.meshgrid(np.linspace(-0.25, 0.25, n_grid),
                                    np.linspace(-np.pi/4, np.pi/4, n_grid), 
                                    np.linspace(-px, px, n_grid))
        X = [-0.25*np.ones(n_grid**3)]
        for xi in [X2, X3, X4]:
            X.append(xi.flatten())
        X = np.array(X)

        x_data = np.zeros([nx, 1])
        u_data = np.zeros([nu, 1])
        for i in range(X.shape[1]):
            if i%10 == 0:
                print("Collect data: ", i)
            x0 = X[:, i]
            x_traj, u_traj, flag = traj_opt(x0)
            if flag:
                x_data = np.hstack((x_data, x_traj[:, :-1]))
                u_data = np.hstack((u_data, u_traj))

        np.save("planar_pusher/data/correct_dynamics/x_data", x_data)
        np.save("planar_pusher/data/correct_dynamics/u_data", u_data)
    else:
        x_data = np.load("planar_pusher/data/correct_dynamics/x_data.npy")
        u_data = np.load("planar_pusher/data/correct_dynamics/u_data.npy")
        
    Z = x2z(x_data).T

    degrees = [deg1, deg2, deg3]
    os.makedirs("planar_pusher/data/correct_dynamics/{}".format(degrees), exist_ok=True)
    prog = MathematicalProgram()
    z = prog.NewIndeterminates(nz)
    u1_lstsq = prog.NewFreePolynomial(Variables(z), deg1)
    u1_decision_variables = np.array(list(u1_lstsq.decision_variables()))
    nu1 = len(u1_decision_variables)
    calc_basis1 = construct_monomial_basis_from_polynomial(u1_lstsq, nu1, z)
    A1 = calc_basis1(Z)
    u1_coeff, res1, _, _ = np.linalg.lstsq(A1, u_data[0])
    print("Residual for u1: ", res1)
    print(np.mean((A1@u1_coeff-u_data[0])**2))
    u1 = construct_polynomial_from_coeff(u1_lstsq, u1_coeff)
    save_polynomial(u1, z, "planar_pusher/data/correct_dynamics/{}/u1.npy".format(degrees))

    u2_lstsq = prog.NewFreePolynomial(Variables(z), deg2)
    u2_decision_variables = np.array(list(u2_lstsq.decision_variables()))
    nu2 = len(u2_decision_variables)
    calc_basis2 = construct_monomial_basis_from_polynomial(u2_lstsq, nu2, z)
    A2 = calc_basis2(Z)
    u2_coeff, res2, _, _ = np.linalg.lstsq(A2, u_data[1])
    print("Residual for u2: ", res2)
    print(np.mean((A2@u2_coeff-u_data[1])**2))
    u2 = construct_polynomial_from_coeff(u2_lstsq, u2_coeff)
    save_polynomial(u2, z, "planar_pusher/data/correct_dynamics/{}/u2.npy".format(degrees))

    u3_lstsq = prog.NewFreePolynomial(Variables(z), deg3)
    u3_decision_variables = np.array(list(u3_lstsq.decision_variables()))
    nu3 = len(u3_decision_variables)
    calc_basis3 = construct_monomial_basis_from_polynomial(u3_lstsq, nu3, z)
    A3 = calc_basis3(Z)
    u3_coeff, res3, _, _ = np.linalg.lstsq(A3, u_data[2])
    print("Residual for u3: ", res3)
    print(np.mean((A3@u3_coeff-u_data[2])**2))
    u3 = construct_polynomial_from_coeff(u3_lstsq, u3_coeff)
    save_polynomial(u3, z, "planar_pusher/data/correct_dynamics/{}/u3.npy".format(degrees))


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
    ax.add_patch(Rectangle((-px, -px), 2*px, 2*px, 0, edgecolor='deeppink', linestyle="--", linewidth=2, facecolor='none', antialiased="True"))
    plt.xlim([-0.35, 0.2])
    plt.ylim([-0.2, 0.35])
    plt.savefig("planar_pusher/figures/trajectory/correct_dynamics/traj_opt_{}_mu_{}.png".format(deg, mu_p))

def simulate_polynomial_controller():
    fit_polynomial_stabilizing_controller(6, 6, 6)
    degrees = list(np.ones(nu, dtype=int) * 4)
    z_var = MakeVectorVariable(nz, "z")
    u = np.zeros(nu, dtype=Expression)
    for i in range(nu):
        u[i] = load_polynomial(z_var, "planar_pusher/data/correct_dynamics/{}/u{}.npy".format(degrees, i+1))
    x0 = np.array([-0.25, 0.25, 0, 0])
    traj, u_traj = simulate_traj_opt(u, z_var, x0)
    plot_traj(traj, degrees[0])

def simulate_value_function(deg):
    z_var = MakeVectorVariable(nz, "z")
    x0 = np.array([-0.25, 0.25, 0, 0])    
    J_star = load_polynomial(z_var, "planar_pusher/data/correct_dynamics/J_lower_deg_{}_mup_{}.pkl".format(deg, mu_p))
    traj, u_traj, J_traj = simulate(J_star, z_var, x0, T=10,  initial_guess=False)
    plot_traj(traj, deg)

    plt.clf()
    plt.plot(J_traj)
    plt.savefig("planar_pusher/figures/trajectory/correct_dynamics/J_{}_mu_{}.png".format(deg, mu_p))

if __name__ == '__main__':
    traj_guess = np.load("planar_pusher/data/traj.npy")
    simulate_value_function(2)
