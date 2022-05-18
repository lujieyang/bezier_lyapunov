import time
from utils import *
from pydrake.all import (ge, Solve, SolverOptions, CommonSolverOption, Polynomial, Variables)
from scipy.integrate import quad
from pydrake.solvers import mathematicalprogram as mp
import pydrake.symbolic as sym
from pydrake.examples.acrobot import AcrobotGeometry, AcrobotPlant, AcrobotParams
from pydrake.symbolic import Expression
import pickle

import matplotlib.pyplot as plt
from matplotlib import cm

def acrobot_setup():
    nz = 6
    nq = 2
    nx = 2 * nq
    nu = 1

    params = AcrobotParams()
    m1 = params.m1()
    m2 = params.m2()
    l1 = params.l1()
    lc1 = params.lc1()
    lc2 = params.lc2()
    I1 = params.Ic1() + m1*lc1**2
    I2 = params.Ic2() + m2*lc2**2
    g = params.gravity()
    B = np.array([[0], [1]])
    # Map from original state to augmented state.
    # Uses sympy to be able to do symbolic integration later on.
    # x = (theta1, theta2, theta1dot, theta2dot)
    # z = (s1, c1, s2, c2, theta1dot, theta2dot)
    x2z = lambda x : np.array([np.sin(x[0]), np.cos(x[0]), np.sin(x[1]), np.cos(x[1]), x[2], x[3]])

    def T(z):
        T = np.zeros([nz, nx])
        T[0, 0] = z[1]
        T[1, 0] = -z[0]
        T[2, 1] = z[3]
        T[3, 1] = -z[2]
        T[4, 2] = 1
        T[5, 3] = 1
        return T

    def Minv(x):
        M = np.array([[I1 + I2 + m2*l1**2 + 2*m2*l1*lc2*np.cos(x[1]),  I2 + m2*l1*lc2*np.cos(x[1])],
        [I2 + m2*l1*lc2*np.cos(x[1]), I2]])
        return np.linalg.inv(M)

    def f(x, Minv, u, T):
        s1 = np.sin(x[0])
        s12 = np.sin(x[0] + x[1])
        s2 = np.sin(x[1])
        qdot = x[nq:]
        f_val = np.zeros(nx, dtype=Expression)
        f_val[:nq] = qdot
        entry = m2*l1*lc2*s2*qdot[1]
        C = np.array([[-2*entry, -entry],
        [m2*l1*lc2*s2*qdot[0], 0]])
        tau_q = np.array([-m1*g*lc1*s1 - m2*g*(l1*s1+lc2*s12),-m2*g*lc2*s12])
        f_val[nq:] = Minv@(tau_q + B@u - C@qdot)
        return T @ f_val
    
    def f2(Minv, T):
        f2_val = np.zeros([nx, nu])
        f2_val[nq:, :] = Minv @ B
        return T@f2_val

    # State limits (region of state space where we approximate the value function).
    x_max = np.array([2*np.pi, np.pi/2, 3, 3])
    x_min = np.array([0, -np.pi/2, -3, -3])

    # Equilibrium point in both the system coordinates.
    x0 = np.array([np.pi, 0, 0, 0])
    z0 = x2z(x0)
    z0[np.abs(z0)<=1e-6] = 0
        
    # Quadratic running cost in augmented state.
    Q = np.diag(np.ones(nz)) * 2
    R = np.diag([1]) 
    def l(z, u):
        return (z - z0).dot(Q).dot(z - z0) + u.dot(R).dot(u)


    Rinv = np.linalg.inv(R)
    params_dict = {"x_min": x_min, "x_max": x_max, "x2z":x2z, "f":f, "l":l,
                   "nz": nz, "f2": f2, "Rinv": Rinv, "z0": z0, "Minv": Minv,
                   "T":T, "nq": nq}
    return params_dict

def convex_sampling_hjb_lower_bound(deg, params_dict, n_mesh=6, objective="", visualize=True):
    print("Objective: ", objective)
    # Sample for nonnegativity constraint of HJB RHS
    nz = params_dict["nz"]
    nq = params_dict["nq"]
    Minv = params_dict["Minv"]
    T = params_dict["T"]
    f = params_dict["f"]
    l = params_dict["l"]
    f2 = params_dict["f2"]
    x2z = params_dict["x2z"]
    z_max = np.array([1, 1, 1, 1, 3, 3])
    z_min = -z_max
    z_min[3] = 0

    prog = MathematicalProgram()
    z = prog.NewIndeterminates(nz, "z")
    J = prog.NewFreePolynomial(Variables(z), deg)
    J_expr = J.ToExpression()
    
    dJdz = J_expr.Jacobian(z)

    mesh_pts = np.linspace(params_dict["x_min"], params_dict["x_max"], n_mesh)

    start_time = time.time()
    J_val = []
    # The order of theta1/2/dot is important!
    for i in range(n_mesh):
        print("Mesh x0 No.", i)
        theta1 = mesh_pts[i, 0]
        for j in range(n_mesh):
            theta2 = mesh_pts[j, 1]
            for k in range(n_mesh):
                thetadot1 = mesh_pts[k, 2]
                for h in range(n_mesh):
                    thetadot2 = mesh_pts[h, 3]
                    x = np.array([theta1, theta2, thetadot1, thetadot2])
                    z_val = x2z(x)
                    z_val[np.abs(z_val)<=1e-6] = 0
                    J_val.append(J_expr.EvaluatePartial(dict(zip(z, z_val))))
                    # RHS of HJB
                    Minv_val = Minv(x)
                    T_val = T(z_val)
                    f2_val = f2(Minv_val, T_val)
                    dJdz_val = np.zeros(nz, dtype=Expression)
                    for n in range(nz): 
                        dJdz_val[n] = dJdz[n].EvaluatePartial(dict(zip(z, z_val)))
                    u_opt = calc_u_opt(dJdz_val, f2_val, params_dict["Rinv"])
                    f_val = f(x, Minv_val, u_opt, T_val)
                    constr = -(l(z_val, u_opt) + dJdz_val.dot(f_val))
                    poly = Polynomial(constr)
                    poly = poly.RemoveTermsWithSmallCoefficients(1e-6)
                    variables, map_var_to_index = sym.ExtractVariablesFromExpression(
                        constr)
                    Q, b, c = sym.DecomposeQuadraticPolynomial(poly, map_var_to_index)
                    try:
                        prog.AddQuadraticAsRotatedLorentzConeConstraint(
                            Q, b, c, variables, psd_tol=1e-5)
                    except:
                        pass
    prog.AddLinearConstraint(ge(np.array(J_val), 0))
    end_time = time.time()
    print("Time for adding constraints: ", end_time-start_time)

    if objective=="integrate_all":
        obj = J
        for i in range(nz):
            obj = obj.Integrate(z[i], z_min[i], z_max[i])
        prog.AddCost(-obj.ToExpression())
    elif objective=="integrate_ring":
        obj = J
        for i in range(2*nq, nz):
            obj = obj.Integrate(z[i], z_min[i], z_max[i])
        c_r = 1
        cost = 0
        for m,coeff in obj.monomial_to_coefficient_map().items(): 
            s1_deg = m.degree(z[0]) 
            c1_deg = m.degree(z[1])
            s2_deg = m.degree(z[2]) 
            c2_deg = m.degree(z[3])
            monomial_int1 = quad(lambda x: np.sin(x)**s1_deg * np.cos(x)**c1_deg, 0, 2*np.pi)[0]
            monomial_int2 = quad(lambda x: np.sin(x)**s2_deg * np.cos(x)**c2_deg, 0, 2*np.pi)[0]
            if np.abs(monomial_int1) <=1e-5:
                monomial_int1 = 0
            if np.abs(monomial_int2) <=1e-5:
                monomial_int2 = 0
            cost += monomial_int1 * monomial_int2 * coeff
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
    Minv = params_dict["Minv"]
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
        Minv_val = Minv(x)
        T_val = T(z_val)
        f2_val = f2(Minv_val, T_val)
        dJdz_val = np.zeros(nz, dtype=Expression)
        for n in range(nz): 
            dJdz_val[n] = dJdz[n].Evaluate(dict(zip(z, z_val)))
        U[i] = calc_u_opt(dJdz_val, f2_val, params_dict["Rinv"])

    fig = plt.figure(figsize=(9, 4))
    ax = fig.subplots()
    ax.set_xlabel("q1")
    ax.set_ylabel("q2")
    ax.set_title("Cost-to-Go")
    im = ax.imshow(J.reshape(X1.shape),
            cmap=cm.jet, aspect='auto',
            extent=(x_min[0], x_max[0], x_min[1], x_max[1]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.savefig("acrobot/figures/{}_{}.png".format(file_name, poly_deg))

    fig = plt.figure(figsize=(9, 4))
    ax = fig.subplots()
    ax.set_xlabel("q1")
    ax.set_ylabel("q2")
    ax.set_title("Policy")
    im = ax.imshow(U.reshape(X1.shape),
            cmap=cm.jet, aspect='auto',
            extent=(x_min[0], x_max[0], x_min[1], x_max[1]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.savefig("acrobot/figures/{}_policy_{}.png".format(file_name, poly_deg))

if __name__ == '__main__':
    poly_deg = 2
    n_mesh = 15
    print("Deg: ", poly_deg)
    params_dict = acrobot_setup()
    J_star, z = convex_sampling_hjb_lower_bound(poly_deg, params_dict, n_mesh=n_mesh, objective="integrate_ring", visualize=True)

    C = extract_polynomial_coeff_dict(J_star, z)
    f = open("acrobot/data/J_{}_{}".format(poly_deg, n_mesh),"wb")
    pickle.dump(C, f)
    f.close()