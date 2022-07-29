import numpy as np
import os
from scipy.integrate import quad
from scipy import integrate
from utils import save_polynomial, load_polynomial
from pydrake.all import (MathematicalProgram, Variables, Expression, Solve, Polynomial, SolverOptions, CommonSolverOption)

import matplotlib.pyplot as plt
from matplotlib import cm

import matplotlib
matplotlib.use('Agg')


def planar_pusher_sos_lower_bound(deg, objective="integrate_ring", visualize=False, test=False):
    """x = [x, y, theta, py]
    z = [x, y, s, c, py]
    u = [fn, ft, v_py]
    """
    nz = 5
    nx = 4
    nu = 3
    
    x0 = np.zeros(4)
    px = 0.05
    mu_g = 0.35
    mu_p = 0.2
    m = 1
    g = 9.81
    fm = mu_g*m*g
    mm = integrate.dblquad(lambda y, x: np.sqrt(x**2+y**2), -px, px, lambda x: -px, lambda x: px)[0]/(2*px)**2 *mu_g * m * g

    d_theta = np.pi/4
    z_max = np.array([0.15, 0.15, np.sin(d_theta), 1, px])
    z_min = np.array([-0.15, -0.15, -np.sin(d_theta), np.cos(d_theta), -px])
    assert (z_min<=z_max).all()

    x2z = lambda x : np.array([x[0], x[1], np.sin(x[2]), np.cos(x[2]), x[3]])
    # Equilibrium point in both the system coordinates.
    z0 = x2z(x0)
        
    # Quadratic running cost in augmented state.
    Q = np.diag([100, 100, 150, 150, 0])
    R = np.eye(nu)/100
    def l_cost(z, u):
        return (z - z0).dot(Q).dot(z - z0) + (u).dot(R).dot(u )

    def f(z, u, dtype=Expression):
        assert len(z) == nz
        assert len(u) == nu
        s = z[2]
        c = z[3]
        f_val = np.zeros(nz, dtype=dtype)
        f_val[0] = (c*u[0] - s*u[1])/fm**2
        f_val[1] = (s*u[0] + c*u[1])/fm**2
        thetadot = (-z[-1]*u[0] + px*u[1])/mm**2  # Dynamics should be -px (px is positive)
        f_val[2] = c*thetadot
        f_val[3] = -s*thetadot
        f_val[4] = u[-1]
        return f_val

    def fx(x, u, dtype=Expression):
        assert len(x) == nx
        assert len(u) == nu
        s = np.sin(x[2])
        c = np.cos(x[2])
        f_val = np.zeros(nx, dtype=dtype)
        f_val[0] = (c*u[0] - s*u[1])/fm**2
        f_val[1] = (s*u[0] + c*u[1])/fm**2
        thetadot = (-x[-1]*u[0] + px*u[1])/mm**2
        f_val[2] = thetadot
        f_val[3] = u[-1]
        return f_val
    
    def f2x(x, dtype=Expression):
        assert len(x) == nx
        s = np.sin(x[2])
        c = np.cos(x[2])
        f2_val = np.zeros([nx, nu], dtype=dtype)
        f2_val[0] = np.array([c, -s, 0])/fm**2
        f2_val[1] = np.array([s, c, 0])/fm**2
        f2_val[2] = np.array([-x[-1], px, 0])/mm**2
        f2_val[3] = np.array([0, 0, 1])
        return f2_val
    
    if test:
        return nz, nu, f, fx, f2x, mu_p, px, l_cost, x2z

    non_q_idx = [0, 1, 4]

    prog = MathematicalProgram()
    z = prog.NewIndeterminates(nz, 'z')
    u = prog.NewIndeterminates(nu, 'u')
    J = prog.NewFreePolynomial(Variables(z), deg)
    J_expr = J.ToExpression()

    # Maximize volume beneath the value function.
    if objective=="integrate_all":
        obj = J
        for i in range(nz):
            obj = obj.Integrate(z[i], z_min[i], z_max[i])
        prog.AddCost(-obj.ToExpression())
    elif objective=="integrate_ring":
        obj = J
        for i in non_q_idx:
            obj = obj.Integrate(z[i], z_min[i], z_max[i])
        cost = 0
        for monomial,coeff in obj.monomial_to_coefficient_map().items(): 
            s1_deg = monomial.degree(z[2]) 
            c1_deg = monomial.degree(z[3])
            monomial_int1 = quad(lambda x: np.sin(x)**s1_deg * np.cos(x)**c1_deg, -d_theta, d_theta)[0]
            cost += monomial_int1 * coeff        
        poly = Polynomial(cost)
        cost_coeff = [c.Evaluate() for c in poly.monomial_to_coefficient_map().values()]
        # Make the numerics better
        cost = cost/np.max(np.abs(cost_coeff))
        cost = Polynomial(cost).RemoveTermsWithSmallCoefficients(1e-6)
        prog.AddLinearCost(-cost.ToExpression())

    # J(z0) = 0.
    J0 = J_expr.EvaluatePartial(dict(zip(z, z0)))
    prog.AddLinearConstraint(J0 == 0)

    # Enforce Bellman inequality.
    f_val = f(z, u)
    J_dot = J_expr.Jacobian(z).dot(f_val)
    LHS = J_dot + l_cost(z, u)

    zu = np.concatenate((z, u))

    lam_deg = Polynomial(LHS).TotalDegree()
    # S procedure for s^2 + c^2 = 1.
    lam_ring = prog.NewFreePolynomial(Variables(z), lam_deg).ToExpression()
    S_ring = lam_ring * (z[2]**2 + z[3]**2 - 1)

    # Friction complementarity constraints
    lam_a = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_deg/2)*2))[0].ToExpression()
    lam_b = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_deg/2)*2))[0].ToExpression()
    lam_c = prog.NewFreePolynomial(Variables(z), lam_deg).ToExpression()
    lam_d = 0 #prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_deg/2)*2))[0].ToExpression()
    S_complementarity = -lam_a*u[-1]*u[1] + lam_b*(u[1]**2 - mu_p**2*u[0]**2) + lam_c*(u[1]**2 - mu_p**2*u[0]**2)*u[-1] -lam_d * u[0] 

    S_Jdot = 0
    for i in np.arange(nz):
        lam = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_deg/2)*2))[0].ToExpression()
        S_Jdot += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
    prog.AddSosConstraint(LHS + S_ring + S_Jdot + S_complementarity)

    # Enforce that value function is PD
    lam_r = prog.NewFreePolynomial(Variables(z), deg).ToExpression()
    S_r = lam_r * (z[2]**2 + z[3]**2 - 1)
    S_J = 0
    for i in np.arange(nz):
        lam = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
        S_J += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
    prog.AddSosConstraint(J_expr + S_J + S_r)

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    assert result.is_success()
    J_star = Polynomial(result.GetSolution(J_expr)).RemoveTermsWithSmallCoefficients(1e-6)

    save_polynomial(J_star, z, 'planar_pusher/data/correct_dynamics/J_lower_deg_{}_mup_{}.pkl'.format(deg, mu_p))
    if visualize:
        plot_value_function(J_star, z, z_max, x2z, file_name="correct_dynamics/lower_bound_constrained_lqr_{}_{}".format(objective, deg), plot_states="xtheta")
    return J_star, z

def plot_value_function(J_star, z, z_max, x2z, file_name="", plot_states="xy", switch_contact=False):
    x_max = np.zeros(4)
    x_max[:2] = z_max[:2]
    x_max[2] = np.pi/2
    x_max[3] = z_max[4]
    x_min = -x_max

    zero_vector = np.zeros(51*51)
    if plot_states == "xtheta":
        y_idx = 2
        X1, THETA = np.meshgrid(np.linspace(x_min[0], x_max[0], 51),
                        np.linspace(x_min[2], x_max[2], 51))
        if switch_contact:
            X = np.vstack((X1.flatten(), zero_vector, THETA.flatten(), zero_vector, zero_vector))
        else:
            X = np.vstack((X1.flatten(), zero_vector, THETA.flatten(), zero_vector))
        ylabel="theta"
    elif plot_states == "xy":
        y_idx = 1
        X1, Y = np.meshgrid(np.linspace(x_min[0], x_max[0], 51),
                        np.linspace(x_min[1], x_max[1], 51))
        if switch_contact:
            X = np.vstack((X1.flatten(), Y.flatten(), zero_vector, zero_vector, zero_vector))
        else:
            X = np.vstack((X1.flatten(), Y.flatten(), zero_vector, zero_vector))
        ylabel="y"
    elif plot_states == "xpx":
        y_idx = -1
        X1, PX = np.meshgrid(np.linspace(x_min[0], x_max[0], 51),
                np.linspace(x_min[-1], x_max[-1], 51))
        X = np.vstack((X1.flatten(), zero_vector, zero_vector, PX.flatten(), zero_vector))        

    Z = x2z(X)
    J = np.zeros(Z.shape[1])
    for i in range(Z.shape[1]):
        z_val = Z[:, i]
        J[i] = J_star.Evaluate(dict(zip(z, z_val)))

    fig = plt.figure()
    ax = fig.subplots()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # ax.set_xlabel("x")
    # ax.set_ylabel(ylabel)
    # ax.set_title("Cost-to-Go")
    im = ax.imshow(J.reshape(X1.shape),
            cmap=cm.jet, aspect='auto',
            extent=(x_min[0], x_max[0], x_max[y_idx], x_min[y_idx]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.savefig("planar_pusher/figures/{}_{}.png".format(file_name, plot_states))

def planar_pusher_4_modes_contact_switching_sos_lower_bound(deg, objective="integrate_ring", eps=1e-3, visualize=False, test=False):
    """x = [x, y, theta, px, py]
    z = [x, y, s, c, px, py]
    u = [fn, ft, v_px, v_py]
    """
    nz = 6
    nx = 5
    nu = 4
    
    x0 = np.zeros(nx)
    px = 0.05
    mu_g = 0.35
    mu_p = 0.3
    m = 1
    g = 9.81
    fm = mu_g*m*g
    mm = integrate.dblquad(lambda y, x: np.sqrt(x**2+y**2), -px, px, lambda x: -px, lambda x: px)[0]/(2*px)**2 *mu_g * m * g

    d_theta = np.pi/4
    z_max = np.array([0.15, 0.15, np.sin(d_theta), 1, px, px])
    z_min = np.array([-0.15, -0.15, -np.sin(d_theta), np.cos(d_theta), -px, -px])
    assert (z_min<=z_max).all()

    x2z = lambda x : np.array([x[0], x[1], np.sin(x[2]), np.cos(x[2]), x[3], x[4]])
    # Equilibrium point in both the system coordinates.
    z0 = x2z(x0)
        
    # Quadratic running cost in augmented state.
    Q = np.diag([100, 100, 150, 150, 0, 0]) * 100
    R = np.eye(nu)/100
    def l_cost(z, u):
        return (z - z0).dot(Q).dot(z - z0) + (u).dot(R).dot(u)
    
    Q_teleport_scale = 1
    Q_teleport = np.eye(2) * Q_teleport_scale
    def l_teleport(z_pre, z_post):
        assert len(z_pre) == 2
        assert len(z_post) == 2
        return (z_pre - z_post).dot(Q_teleport).dot(z_pre - z_post)

    def f(z, u, n, d, dtype=Expression):
        assert len(z) == nz
        assert len(u) == nu
        s = z[2]
        c = z[3]
        f_val = np.zeros(nz, dtype=dtype)
        R = np.array([[c, -s, 0],
                      [s, c, 0],
                      [0, 0, 1]])
        L = np.diag([1/fm**2, 1/fm**2, 1/mm**2])
        J = np.array([[1, 0, -z[5]],
                      [0, 1, z[4]]])
        f_val[:3] = R@L@J.T@(n*u[0]+d*u[1])
        theta_dot = np.copy(f_val[2])
        f_val[2] = c*theta_dot
        f_val[3] = -s*theta_dot
        f_val[4] = u[2]
        f_val[5] = u[3]
        return f_val

    def fx(x, u, n, d, dtype=Expression):
        assert len(x) == nx
        assert len(u) == nu
        s = np.sin(x[2])
        c = np.cos(x[2])
        f_val = np.zeros(nx, dtype=dtype)
        R = np.array([[c, -s, 0],
                      [s, c, 0],
                      [0, 0, 1]])
        L = np.diag([1/fm**2, 1/fm**2, 1/mm**2])
        J = np.array([[1, 0, -x[-1]],
                      [0, 1, x[-2]]])
        f_val[:3] = R@L@J.T@(n*u[0]+d*u[1])
        f_val[3] = u[2]
        f_val[4] = u[3]
        return f_val
    
    def f2x(x, n, d, dtype=Expression):
        assert len(x) == nx
        f2_val = np.zeros([nx, nu], dtype=dtype)
        f1_val = fx(x, np.zeros(nu), n, d)
        for i in range(nu):
            ue = np.zeros(nu)
            ue[i] = 1
            f2_val[:, i] = fx(x, ue, n, d) - f1_val
        return f2_val
    
    if test:
        return nz, nu, f, fx, f2x, mu_p, px, l_cost, l_teleport, x2z

    non_q_idx = [0, 1, 4, 5]

    prog = MathematicalProgram()
    z = prog.NewIndeterminates(nz, 'z')
    u = prog.NewIndeterminates(nu, 'u')
    J = prog.NewFreePolynomial(Variables(z), deg)
    J_expr = J.ToExpression()

    # Maximize volume beneath the value function.
    if objective=="integrate_all":
        obj = J
        for i in range(nz):
            obj = obj.Integrate(z[i], z_min[i], z_max[i])
        prog.AddCost(-obj.ToExpression())
    elif objective=="integrate_ring":
        obj = J
        for i in non_q_idx:
            obj = obj.Integrate(z[i], z_min[i], z_max[i])
        cost = 0
        for monomial,coeff in obj.monomial_to_coefficient_map().items(): 
            s1_deg = monomial.degree(z[2]) 
            c1_deg = monomial.degree(z[3])
            monomial_int1 = quad(lambda x: np.sin(x)**s1_deg * np.cos(x)**c1_deg, -d_theta, d_theta)[0]
            cost += monomial_int1 * coeff        
        poly = Polynomial(cost)
        cost_coeff = [c.Evaluate() for c in poly.monomial_to_coefficient_map().values()]
        # Make the numerics better
        cost = cost/np.max(np.abs(cost_coeff))
        cost = Polynomial(cost).RemoveTermsWithSmallCoefficients(1e-6)
        prog.AddLinearCost(-cost.ToExpression())

    # J(z0) = 0.
    J0 = J_expr.EvaluatePartial(dict(zip(z, z0)))
    prog.AddLinearConstraint(J0 == 0)

    # Enforce Bellman inequality.
    normals = [np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])]
    tangentials = [np.array([0, 1]), np.array([0, -1]), np.array([-1, 0]), np.array([1, 0])]
    for j in range(4):
        n = normals[j]
        d = tangentials[j]
        f_val = f(z, u, n, d)
        J_dot = J_expr.Jacobian(z).dot(f_val)
        LHS = J_dot + l_cost(z, u)

        zu = np.concatenate((z, u))

        lam_deg = Polynomial(LHS).TotalDegree()
        # S procedure for s^2 + c^2 = 1.
        lam_ring = prog.NewFreePolynomial(Variables(z), lam_deg).ToExpression()
        S_ring = lam_ring * (z[2]**2 + z[3]**2 - 1)

        # Friction complementarity constraints
        fn = u[0]
        ft = u[1]
        v_px = u[2]
        v_py = u[3]
        if i<=1:
            # On the left and right surface
            lam_a = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_deg/2)*2))[0].ToExpression()
            lam_b = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_deg/2)*2))[0].ToExpression()
            lam_c = prog.NewFreePolynomial(Variables(z), lam_deg).ToExpression()
            lam_d = prog.NewFreePolynomial(Variables(z), lam_deg).ToExpression()
            S_complementarity = -lam_a*v_py*ft + lam_b*(ft**2 - mu_p**2*fn**2) + lam_c*(ft**2 - mu_p**2*fn**2)*v_py -lam_d * v_px
            int_idx = [0, 1, 2, 3, 5]
        else:
            # On the top and bottom surface
            lam_a = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_deg/2)*2))[0].ToExpression()
            lam_b = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_deg/2)*2))[0].ToExpression()
            lam_c = prog.NewFreePolynomial(Variables(z), lam_deg).ToExpression()
            lam_d = prog.NewFreePolynomial(Variables(z), lam_deg).ToExpression()
            S_complementarity = -lam_a*v_px*ft + lam_b*(ft**2 - mu_p**2*fn**2) + lam_c*(ft**2 - mu_p**2*fn**2)*v_px -lam_d * v_py
            int_idx = np.arange(5)

        # Restrain px, py to be on the surface
        lam_s = prog.NewFreePolynomial(Variables(z), lam_deg).ToExpression()
        S_s = lam_s * (n.dot(z[-2:]) + px)

        S_Jdot = 0
        for i in int_idx:
            lam = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_deg/2)*2))[0].ToExpression()
            S_Jdot += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
        prog.AddSosConstraint(LHS + S_ring + S_Jdot + S_complementarity + S_s)
    
        # Teleportation
        z_post = prog.NewIndeterminates(2, 'z_post')
        J_post = J_expr.Substitute(dict(zip(z[4:], z_post)))
        S_teleport = 0
        for i in int_idx:
            lam = prog.NewSosPolynomial(Variables(z), deg-2)[0].ToExpression()
            S_teleport += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
        for i in np.arange(2):
            lam = prog.NewSosPolynomial(Variables(z), deg-2)[0].ToExpression()
            S_teleport += lam*(z_post[i]-z_max[i+4])*(z_post[i]-z_min[i+4])
        # Restrain z_post to be on a different surface from z
        lam = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
        S_teleport += lam*(-n.dot(np.array([z_post[0], z_post[1]])))  # n.dot(p) >=0
        # Restrain z_post to be on the surface
        lam_s = prog.NewFreePolynomial(Variables(z), deg-2).ToExpression()
        S_s = (z_post[0]-px)*(z_post[0]+px)*(z_post[1]-px)*(z_post[1]+px)
        lam_r = prog.NewFreePolynomial(Variables(z), lam_deg).ToExpression()
        S_r = lam_r * (z[2]**2 + z[3]**2 - 1)
        prog.AddSosConstraint(l_teleport(z[4:], z_post) + J_post - J_expr + S_teleport + S_s + S_r)

    # Enforce that value function is PD
    lam_r = prog.NewFreePolynomial(Variables(z), deg).ToExpression()
    S_r = lam_r * (z[2]**2 + z[3]**2 - 1)
    S_J = 0
    for i in np.arange(nz):
        lam = prog.NewSosPolynomial(Variables(z), deg-2)[0].ToExpression()
        S_J += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
    prog.AddSosConstraint(J_expr + S_J + S_r)

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    assert result.is_success()
    J_star = Polynomial(result.GetSolution(J_expr)).RemoveTermsWithSmallCoefficients(1e-6)

    os.makedirs("planar_pusher/data/four_modes/{}".format(Q_teleport_scale), exist_ok=True)
    os.makedirs("planar_pusher/figures/four_modes/{}".format(Q_teleport_scale), exist_ok=True)

    save_polynomial(J_star, z, 'planar_pusher/data/four_modes/{}/J_lower_deg_{}_100.pkl'.format(Q_teleport_scale, deg))
    if visualize:
        plot_value_function(J_star, z, z_max, x2z, file_name="four_modes/{}/lower_bound_constrained_lqr_{}_{}_100".format(Q_teleport_scale, objective, deg), plot_states="xtheta", switch_contact=True)
    return J_star, z

if __name__ == '__main__':
    deg = 2
    J_star, z = planar_pusher_4_modes_contact_switching_sos_lower_bound(deg, objective="integrate_ring",visualize=True)