import time
from pendulum_swingup.utils import *
from pydrake.all import (Solve, SolverOptions, CommonSolverOption, Polynomial, Variables)
from scipy.integrate import quad
from pydrake.solvers import mathematicalprogram as mp
import pydrake.symbolic as sym
from pydrake.symbolic import Expression

import matplotlib.pyplot as plt
from matplotlib import cm

def cartpole_setup():
    nz = 5
    nq = 2
    nx = 2 * nq
    nu = 1

    mc = 10
    mp = 1
    l = .5
    g = 9.81
    B = np.array([[0], [1]])
    # Map from original state to augmented state.
    # Uses sympy to be able to do symbolic integration later on.
    # x = (x, theta, xdot, thetadot)
    # z = (x, s, c, xdot, thetadot)
    x2z = lambda x : np.array([x[0], np.sin(x[1]), np.cos(x[1]), x[2], x[3]])

    def T(z):
        T = np.zeros([nz, nx])
        T[0, 0] = 1
        T[1, 1] = z[2]
        T[2, 1] = -z[1]
        T[3, 2] = 1
        T[4, 3] = 1
        return T

    def f(x, u, T):
        s = np.sin(x[1])
        c = np.cos(x[1])
        qdot = x[nq:]
        f_val = np.zeros(nx, dtype=Expression)
        f_val[:nq] = qdot
        f_val[2] = (u + mp*s*(l*qdot[1]**2+g*c))/(mc+mp*s**2)
        f_val[3] = (-u*c - mp*l*qdot[1]**2*c*s - (mc+mp)*g*s)/(mc+mp*s**2)/l
        return T @ f_val
    
    def f2(x, T):
        s = np.sin(x[1])
        c = np.cos(x[1])
        f2_val = np.zeros([nx, nu])
        f2_val[2, :] = 1/(mc+mp*s**2)
        f2_val[3, :] =-c/(mc+mp*s**2)/l
        return T@f2_val

    # State limits (region of state space where we approximate the value function).
    x_max = np.array([2, 2*np.pi, 3, 3])
    x_min = np.array([-2, 0, -3, -3])

    # Equilibrium point in both the system coordinates.
    x0 = np.array([0, np.pi, 0, 0])
    z0 = x2z(x0)
    z0[np.abs(z0)<=1e-6] = 0
        
    # Quadratic running cost in augmented state.
    Q = np.diag([10, 10, 10, 1, 1])
    R = np.diag([1]) 
    def l_cost(z, u):
        return (z - z0).dot(Q).dot(z - z0) + u.dot(R).dot(u)


    Rinv = np.linalg.inv(R)
    params_dict = {"x_min": x_min, "x_max": x_max, "x2z":x2z, "f":f, "l_cost":l_cost,
                   "nz": nz, "f2": f2, "Rinv": Rinv, "z0": z0, "T":T, "nq": nq}
    return params_dict

def convex_sampling_hjb_lower_bound(deg, params_dict, n_mesh=6, objective="", visualize=True):
    print("Objective: ", objective)
    # Sample for nonnegativity constraint of HJB RHS
    nz = params_dict["nz"]
    T = params_dict["T"]
    f = params_dict["f"]
    l_cost = params_dict["l_cost"]
    f2 = params_dict["f2"]
    x2z = params_dict["x2z"]
    z_max = np.array([2, 1, 1, 3, 3])
    z_min = -z_max

    prog = MathematicalProgram()
    z = prog.NewIndeterminates(nz, "z")
    J = prog.NewFreePolynomial(Variables(z), deg)
    J_expr = J.ToExpression()
    
    dJdz = J_expr.Jacobian(z)

    mesh_pts = np.linspace(params_dict["x_min"], params_dict["x_max"], n_mesh)

    start_time = time.time()
    for i in range(n_mesh):
        print("Mesh x0 No.", i)
        position = mesh_pts[i, 0]
        for j in range(n_mesh):
            theta = mesh_pts[j, 1]
            for k in range(n_mesh):
                xdot = mesh_pts[k, 2]
                for h in range(n_mesh):
                    thetadot = mesh_pts[h, 3]
                    x = np.array([position, theta, xdot, thetadot])
                    z_val = x2z(x)
                    z_val[np.abs(z_val)<=1e-6] = 0
                    prog.AddLinearConstraint(J_expr.EvaluatePartial(dict(zip(z, z_val))) >= 0)
                    # RHS of HJB
                    T_val = T(z_val)
                    f2_val = f2(x, T_val)
                    dJdz_val = np.zeros(nz, dtype=Expression)
                    for n in range(nz): 
                        dJdz_val[n] = dJdz[n].EvaluatePartial(dict(zip(z, z_val)))
                    u_opt = calc_u_opt(dJdz_val, f2_val, params_dict["Rinv"])
                    f_val = f(x, u_opt, T_val)
                    constr = -(l_cost(z_val, u_opt) + dJdz_val.dot(f_val))
                    poly = Polynomial(constr[0])
                    poly = poly.RemoveTermsWithSmallCoefficients(1e-6)
                    variables, map_var_to_index = sym.ExtractVariablesFromExpression(
                        constr)
                    Q, b, c = sym.DecomposeQuadraticPolynomial(poly, map_var_to_index)
                    try:
                        prog.AddQuadraticAsRotatedLorentzConeConstraint(
                            Q, b, c, variables, psd_tol=1e-5)
                    except:
                        pass
    end_time = time.time()
    print("Time for adding constraints: ", end_time-start_time)

    if objective=="integrate_all":
        obj = J
        for i in range(nz):
            obj = obj.Integrate(z[i], z_min[i], z_max[i])
        prog.AddCost(-obj.ToExpression())
    elif objective=="integrate_ring":
        obj = J
        for i in range(3, nz):
            obj = obj.Integrate(z[i], z_min[i], z_max[i])
        c_r = 1
        cost = 0
        for monomial,coeff in obj.monomial_to_coefficient_map().items(): 
            s1_deg = monomial.degree(z[1]) 
            c1_deg = monomial.degree(z[2])
            monomial_int1 = quad(lambda x: np.sin(x)**s1_deg * np.cos(x)**c1_deg, 0, 2*np.pi)[0]
            if np.abs(monomial_int1) <=1e-5:
                monomial_int1 = 0
            cost += monomial_int1 * coeff
        prog.AddLinearCost(-c_r * cost)

    # J(z0) = 0
    J0 = J_expr.EvaluatePartial(dict(zip(z, params_dict["z0"])))
    prog.AddLinearConstraint(J0 == 0)
    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)

    print("="*10, "Solving","="*20)
    solve_start = time.time()
    result = Solve(prog)
    solve_end = time.time()
    print("Time for solving: ", solve_end-solve_start)

    J_star = result.GetSolution(J)

    # assert result.is_success()
    print(result.is_success())
    print("area: ", result.get_optimal_cost())
    print("# of quadratic constraints: ", len(
        prog.rotated_lorentz_cone_constraints()))

    dJdz = J_star.ToExpression().Jacobian(z)
    if visualize:
        plot_value_function(J_star, z, params_dict, deg,
        file_name="convex_sampling_hjb_lower_bound_{}_mesh_{}".format(objective, n_mesh))

    return J_star, z

def plot_value_function(J_star, z, params_dict, poly_deg, file_name=""):
    nz = params_dict["nz"]
    x_min = params_dict["x_min"]
    x_max = params_dict["x_max"]
    T = params_dict["T"]
    f2 = params_dict["f2"]
    x2z = params_dict["x2z"]

    dJdz = J_star.ToExpression().Jacobian(z)

    X1, X2 = np.meshgrid(np.linspace(x_min[0], x_max[0], 51),
                    np.linspace(x_min[1], x_max[1], 51))
    X = np.vstack((X1.flatten(), X2.flatten(), np.zeros(51*51), np.zeros(51*51)))
    Z = x2z(X)
    J = np.zeros(Z.shape[1])
    U = np.zeros(Z.shape[1])
    for i in range(Z.shape[1]):
        z_val = Z[:, i]
        x = X[:, i]
        J[i] = J_star.Evaluate(dict(zip(z, z_val)))
        T_val = T(z_val)
        f2_val = f2(x, T_val)
        dJdz_val = np.zeros(nz, dtype=Expression)
        for n in range(nz): 
            dJdz_val[n] = dJdz[n].Evaluate(dict(zip(z, z_val)))
        U[i] = calc_u_opt(dJdz_val, f2_val, params_dict["Rinv"])

    fig = plt.figure(figsize=(9, 4))
    ax = fig.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("theta")
    ax.set_title("Cost-to-Go")
    im = ax.imshow(J.reshape(X1.shape),
            cmap=cm.jet, aspect='auto',
            extent=(x_min[0], x_max[0], x_min[1], x_max[1]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.savefig("cartpole/figures/{}_{}.png".format(file_name, poly_deg))

    fig = plt.figure(figsize=(9, 4))
    ax = fig.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("theta")
    ax.set_title("Policy")
    im = ax.imshow(U.reshape(X1.shape),
            cmap=cm.jet, aspect='auto',
            extent=(x_min[0], x_max[0], x_min[1], x_max[1]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.savefig("cartpole/figures/{}_policy_{}.png".format(file_name, poly_deg))

if __name__ == '__main__':
    poly_deg = 2
    print("Deg: ", poly_deg)
    params_dict = cartpole_setup()
    convex_sampling_hjb_lower_bound(poly_deg, params_dict, n_mesh=11, objective="integrate_ring")