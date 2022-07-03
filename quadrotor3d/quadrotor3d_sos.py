import numpy as np
import math
from scipy.spatial.transform import Rotation
from scipy.integrate import quad
import mcint
from utils import extract_polynomial_coeff_dict, calc_u_opt 
import pickle
from pydrake.all import (MathematicalProgram, Variables, Expression, Solve, Polynomial, SolverOptions, CommonSolverOption, 
ControllabilityMatrix, LinearQuadraticRegulator, sin, cos, Linearize)
from pydrake.examples.quadrotor_trig import (QuadrotorTrigPlant)

import matplotlib.pyplot as plt
from matplotlib import cm

import matplotlib
matplotlib.use('Agg')

# Given the degree for the approximate value function and the polynomials
# in the S procedure, solves the SOS and returns the approximate value function
# (together with the objective of the SOS program).
def quadrotor2d_sos_lower_bound(deg, objective="integrate_ring", visualize=False, test=False):
    nz = 7
    nx = 6
    nu = 2
    
    quadrotor = QuadrotorTrigPlant()
    m = quadrotor.mass
    g = quadrotor.gravity
    r = quadrotor.length
    I = quadrotor.inertia
    u0 = m * g / 2. * np.array([1, 1])
    # Map from original state to augmented state.
    # Uses sympy to be able to do symbolic integration later on.
    # x = (x, y, theta, xdot, ydot, thetadot)
    # z = (x, y, s, c, xdot, ydot, thetadot)
    x2z = lambda x : np.array([x[0], x[1], np.sin(x[2]), np.cos(x[2]), x[3], x[4], x[5]])

    def f(z, u, dtype=Expression):
        assert len(z) == nz
        assert len(u) == nu
        s = z[2]
        c = z[3]
        thetadot = z[-1]
        f_val = np.zeros(nz, dtype=dtype)
        f_val[:2] = z[4:6]
        f_val[2] = thetadot * c
        f_val[3] = -thetadot * s
        f_val[4] = -s/m*(u[0]+u[1])
        f_val[5] = c/m*(u[0]+u[1])-g
        f_val[6] = r/I*(u[0]-u[1])
        return f_val

    def f2(z, dtype=Expression):
        assert len(z) == nz
        s = z[2]
        c = z[3]
        f2_val = np.zeros([nz, nu], dtype=dtype)
        f2_val[4,:] = -s/m*np.ones(nu)
        f2_val[5,:] = c/m*np.ones(nu)
        f2_val[6,:] = r/I*np.array([1, -1])
        return f2_val
    
    # State limits (region of state space where we approximate the value function).
    z_max = np.array([1, 1, 1, 1, 4.5, 4.5, 3])
    z_min = np.array([-1, -1, -1, -1, -4.5, -4.5, -3])

    # Equilibrium point in both the system coordinates.
    x0 = np.zeros(nx)
    z0 = x2z(x0)
    z0[np.abs(z0)<=1e-6] = 0
        
    # Quadratic running cost in augmented state.
    Q = np.diag([10, 10, 10, 10, 1, 1, r/(2*np.pi)]) * 1e2
    R = np.array([[0.1, 0.05], [0.05, 0.1]])
    def l_cost(z, u):
        return (z - z0).dot(Q).dot(z - z0) + u.dot(R).dot(u)

    Rinv = np.linalg.inv(R)

    if test:
        return nz, f, f2, x2z, Rinv

    non_q_idx = [0, 1, 4, 5, 6]

    # Set up optimization.
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
        c_r = 1
        cost = 0
        for monomial,coeff in obj.monomial_to_coefficient_map().items(): 
            s1_deg = monomial.degree(z[2]) 
            c1_deg = monomial.degree(z[3])
            monomial_int1 = quad(lambda x: np.sin(x)**s1_deg * np.cos(x)**c1_deg, -np.pi, np.pi)[0]
            if np.abs(monomial_int1) <=1e-5:
                monomial_int1 = 0
            cost += monomial_int1 * coeff
        poly = Polynomial(cost)
        cost_coeff = [c.Evaluate() for c in poly.monomial_to_coefficient_map().values()]
        # Make the numerics better
        prog.AddLinearCost(-c_r * cost/np.max(np.abs(cost_coeff)))
        # prog.AddLinearCost(-c_r * cost)
    # prog.AddQuadraticCost(1e-3*np.sum(np.array(list(J.decision_variables()))**2), is_convex=True)

    # S procedure for s^2 + c^2 = 1.
    lam = prog.NewFreePolynomial(Variables(z), deg).ToExpression()
    S_ring = lam * (z[2]**2 + z[3]**2 - 1)
    
    # Enforce Bellman inequality.
    f_val = f(z, u)
    J_dot = J_expr.Jacobian(z).dot(f_val)
    if deg <= 0:
        S_Jdot = 0
        for i in non_q_idx:
            lam = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
            S_Jdot += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
        prog.AddSosConstraint(J_dot + l_cost(z, u) + S_ring + S_Jdot)
    else:
        prog.AddSosConstraint(J_dot + l_cost(z, u) + S_ring)

    # Enforce that value function is PD
    if deg <= 0:
        S_J = 0
        for i in non_q_idx:
            lam = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
            S_J += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
        prog.AddSosConstraint(J_expr + S_J)
    else:
        prog.AddSosConstraint(J_expr)

    # J(z0) = 0.
    J0 = J_expr.EvaluatePartial(dict(zip(z, z0)))
    prog.AddLinearConstraint(J0 == 0)

    # Solve and retrieve result.
    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    assert result.is_success()
    J_star = Polynomial(result.GetSolution(J_expr)).RemoveTermsWithSmallCoefficients(1e-6)

    if visualize:
        plot_value_function(J_star, z, z_max, u0, file_name="lower_bound_{}_{}".format(objective, deg), plot_states="xy", u_index=0)
    return J_star, z

def plot_value_function(J_star, u_star, z, z_max, z_min, u0, file_name="", plot_states="xy", u_index=0):
    zero_vector = np.zeros([51*51, 1])
    qwxyz = np.zeros([51*51, 4])
    xyzdot_w = np.zeros([51*51, 6])
    if plot_states == "xz":
        y_limit_idx = 6
        X, Z1 = np.meshgrid(np.linspace(z_min[4], z_max[4], 51),
                        np.linspace(z_min[6], z_max[6], 51))
        Z = np.hstack((qwxyz, X.flatten().reshape(51*51, 1), zero_vector, Z1.flatten().reshape(51*51, 1), xyzdot_w))
        ylabel="z"
    elif plot_states == "xy":
        y_limit_idx = 5
        X, Y = np.meshgrid(np.linspace(z_min[4], z_max[4], 51),
                        np.linspace(z_min[5], z_max[5], 51))
        Z = np.hstack((qwxyz, X.flatten().reshape(51*51, 1), Y.flatten().reshape(51*51, 1), zero_vector, xyzdot_w))
        ylabel="y"

    J = np.zeros(Z.shape[0])
    U = np.zeros(Z.shape[0])
    for i in range(Z.shape[0]):
        z_val = np.squeeze(Z[i])
        J[i] = J_star.Evaluate(dict(zip(z, z_val)))
        U[i] = u_star[u_index].Evaluate(dict(zip(z, z_val))) + u0[u_index]

    fig = plt.figure(figsize=(9, 4))
    ax = fig.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.set_title("Cost-to-Go")
    im = ax.imshow(J.reshape(X.shape),
            cmap=cm.jet, aspect='auto',
            extent=(z_min[4], z_max[4], z_max[y_limit_idx], z_min[y_limit_idx]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.savefig("quadrotor3d/figures/{}_{}.png".format(file_name, plot_states))

    fig = plt.figure(figsize=(9, 4))
    ax = fig.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.set_title("Policy")
    im = ax.imshow(U.reshape(X.shape),
            cmap=cm.jet, aspect='auto',
            extent=(z_min[4], z_max[4], z_max[y_limit_idx], z_min[y_limit_idx]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.savefig("quadrotor3d/figures/{}_policy_{}_u{}.png".format(file_name, plot_states, u_index+1))

def quadrotor3d_constrained_lqr(nz=13, nu=4):
    quadrotor = QuadrotorTrigPlant()
    m = quadrotor.m()
    g = quadrotor.g()
    L = 0.15
    I = np.array([[0.0015, 0, 0], 
          [0, 0.0025, 0], 
          [0, 0, 0.0035]])
    kF = 1
    kM = 0.0245
    A = np.zeros([nz, nz])
    B = np.zeros([nz, nu])
    A[1, 8] = -2*g*kF
    A[2, 7] = 2*g*kF
    A[7, 4] = 1
    A[8, 5] = 1
    A[9, 6] = 1
    A[10, 1] = .5
    A[11, 2] = .5
    A[12, 3] = .5
    B[0, -4:] = np.array([kF/m, 0, -400*kF*L, 285.714*kM])
    B[1, -4:] = np.array([kF/m, 666.667*kF*L, 0, -285.714*kM])
    B[2, -4:] = np.array([kF/m, 0, 400*kF*L, 285.714*kM])
    B[3, -4:] = np.array([kF/m, -666.667*kF*L, 0, -285.714*kM])
    F = np.zeros(nz)
    F[0] = 1
    Q = np.diag([1, 1, 1, 1, 5, 5, 5, 10, 10, 10, 10, 10, 10])
    R = np.eye(nu)
    K, S = LinearQuadraticRegulator(A, B, Q, R, F=F.reshape(1, nz))
    return K

def quadrotor3d_trig_constrained_lqr(nz=13, nu=4):
    quadrotor = QuadrotorTrigPlant()
    context = quadrotor.CreateDefaultContext()
    context.SetContinuousState(np.zeros(nz))
    m = quadrotor.m()
    g = quadrotor.g()
    u0 = m*g/nu * np.ones(nu)
    quadrotor.get_input_port().FixValue(context, u0)
    linearized_quadrotor = Linearize(quadrotor, context)
    F = np.zeros(nz)
    F[0] = 1
    Q = np.diag([1, 1, 1, 1, 5, 5, 5, 10, 10, 10, 10, 10, 10])
    R = np.eye(nu)
    K, S = LinearQuadraticRegulator(linearized_quadrotor.A(), linearized_quadrotor.B(),\
        Q, R, F=F.reshape(1, nz))
    return K, S

def quadrotor3d_sos_upper_bound(deg, deg_lower=0, objective="integrate_ring", visualize=False, test=False, actuator_saturate=False):
    nz = 13
    nu = 4
    
    quadrotor = QuadrotorTrigPlant()
    m = quadrotor.m()
    g = quadrotor.g()
    L = 0.15
    I = np.array([[0.0015, 0, 0], 
          [0, 0.0025, 0], 
          [0, 0, 0.0035]])
    kF = 1
    kM = 0.0245

    u0 = m*g/4 * np.ones(nu)
    u_max = 2.5 * u0
    u_min = np.zeros(4)

    # Map from original state to augmented state.
    # z = (qw-1, qx, qy, qz, x, y, z, xdot, ydot, zdot, omega)

    def f(z, u, dtype=Expression):
        assert len(z) == nz
        assert len(u) == nu
        if dtype == float:
            assert (z[0]+1)**2 + z[1]**2 + z[2]**2 + z[3]**2 == 1
        uF_Bz = kF * u

        Faero_B = np.array([0, 0, uF_Bz.sum()])
        Mx = L * (uF_Bz[1] - uF_Bz[3])
        My = L * (uF_Bz[2] - uF_Bz[0])
        uTau_Bz = kM * u
        Mz = uTau_Bz[0] - uTau_Bz[1] + uTau_Bz[2] - uTau_Bz[3]

        Tau_B= np.array([Mx, My, Mz])
        Fgravity_N = np.array([0, 0, -m * g])

        ## CAUTION: z0 = qw-1
        quat = np.array([z[0]+1, z[1], z[2], z[3]])
        w_NB_B = z[-3:]
        quat_dot = np.zeros(4, dtype=dtype)
        quat_dot[0] = 0.5 * (-w_NB_B[0] * quat[1] - w_NB_B[1] * quat[2] - w_NB_B[2] * quat[3])
        quat_dot[1] = 0.5 * (w_NB_B[0] * quat[0] + w_NB_B[2] * quat[2] - w_NB_B[1] * quat[3])
        quat_dot[2] = 0.5 * (w_NB_B[1] * quat[0] - w_NB_B[2] * quat[1] + w_NB_B[0] * quat[3])
        quat_dot[3] = 0.5 * (w_NB_B[2] * quat[0] + w_NB_B[1] * quat[1] - w_NB_B[0] * quat[2])

        R_NB = np.zeros([3, 3], dtype=dtype)
        R_NB[0, 0] = 1 - 2 * quat[2] * quat[2] - 2 * quat[3] * quat[3]
        R_NB[0, 1] = 2 * quat[1] * quat[2] - 2 * quat[0] * quat[3]
        R_NB[0, 2] = 2 * quat[1] * quat[3] + 2 * quat[0] * quat[2]
        R_NB[1, 0] = 2 * quat[1] * quat[2] + 2 * quat[0] * quat[3]
        R_NB[1, 1] = 1 - 2 * quat[1] * quat[1] - 2 * quat[3] * quat[3]
        R_NB[1, 2] = 2 * quat[2] * quat[3] - 2 * quat[0] * quat[1]
        R_NB[2, 0] = 2 * quat[1] * quat[3] - 2 * quat[0] * quat[2]
        R_NB[2, 1] = 2 * quat[2] * quat[3] + 2 * quat[0] * quat[1]
        R_NB[2, 2] = 1 - 2 * quat[1] * quat[1] - 2 * quat[2] * quat[2]

        xyzDDt = (Fgravity_N + R_NB @ Faero_B) / m

        I_w_NB_B = I @ w_NB_B
        wIw = np.array([w_NB_B[1]*I_w_NB_B[2] - w_NB_B[2]*I_w_NB_B[1], -(w_NB_B[0]*I_w_NB_B[2] - w_NB_B[2]*I_w_NB_B[0]),
        w_NB_B[0]*I_w_NB_B[1] - w_NB_B[1]*I_w_NB_B[0]])
        alpha_NB_B = np.array([(Tau_B[0] - wIw[0]) / I[0, 0],
                                    (Tau_B[1] - wIw[1]) / I[1, 1],
                                    (Tau_B[2] - wIw[2]) / I[2, 2]])
        return np.hstack((quat_dot, z[7:10], xyzDDt, alpha_NB_B)) 


    def f2(z, dtype=Expression):
        assert len(z) == nz
        f2_val = np.zeros([nz, nu], dtype=dtype)
        f1_val = f(z, np.zeros(nu))
        for i in range(nu):
            ue = np.zeros(nu)
            ue[i] = 1
            f2_val[:, i] = f(z, ue) - f1_val
        return f2_val
    
    # State limits (region of state space where we approximate the value function).
    rpy_up = np.array([np.pi, 0.4 * np.pi, np.pi])
    rpy_lo = -rpy_up
    rpys = np.linspace(rpy_lo, rpy_up)
    r = Rotation.from_euler("zyx", rpys)
    r = r.as_quat()
    z_max = np.ones(nz)
    z_max[0] = 0
    xyzw_max = np.max(r, 0)
    z_max[1: 4] = xyzw_max[:3]
    z_max[7:] = 2
    z_min = -z_max
    xyzw_min = np.min(r, 0)
    z_min[0] = xyzw_min[-1] - 1
    z_min[1: 4] = xyzw_min[:3]
    assert (z_max >= z_min).all()

    # Equilibrium point in both the system coordinates.
    z0 = np.zeros(nz)
        
    # Quadratic running cost in augmented state.
    Q = np.diag([1, 1, 1, 1, 5, 5, 5, 10, 10, 10, 10, 10, 10])
    R = np.eye(nu)
    def l_cost(z, u):
        return (z - z0).dot(Q).dot(z - z0) + (u- u0).dot(R).dot(u - u0)

    Rinv = np.linalg.inv(R)

    if test:
        return nz, f, f2, z_max, z_min

    q_idx = np.arange(4)
    non_q_idx = np.arange(4, nz)

    # Set up optimization.        
    prog = MathematicalProgram()
    z = prog.NewIndeterminates(nz, 'z')
    K, _ = quadrotor.SynthesizeTrigLqr()
    u_fixed = -K @ (z-z0) + u0
    J = prog.NewFreePolynomial(Variables(z), deg)
    J_expr = J.ToExpression()

    a = prog.NewSosPolynomial(Variables(z), deg)[0]

    def integrate_quaternion_symbolic(qw_deg, qx_deg, qy_deg, qz_deg):
        rpy = prog.NewIndeterminates(3, "rpy")
        cy = cos(rpy[2]/2)
        sy = sin(rpy[2]/2)
        cp = cos(rpy[1]/2)
        sp = sin(rpy[1]/2)
        cr = cos(rpy[0]/2)
        sr = sin(rpy[0]/2)
        qw = cr * cp * cy + sr * sp * sy - 1
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        q_int = qw**qw_deg * qx**qx_deg * qy**qy_deg * qz**qz_deg
        for i in np.arange(3):
            q_int = q_int.Integrate(rpy[i], rpy_lo[i], rpy_up[i])
        return q_int.ToExpression()
    
    def sampler():
        while True:
            r = np.random.uniform(rpy_lo[0], rpy_up[0])
            p = np.random.uniform(rpy_lo[1], rpy_up[1])
            y = np.random.uniform(rpy_lo[2], rpy_up[2])
            yield (r, p, y)

    def integrate_quaternion_monte_carlo(qw_deg, qx_deg, qy_deg, qz_deg, n_samples=10000):
        def integrand(x):
            cy = math.cos(x[2]/2)
            sy = math.sin(x[2]/2)
            cp = math.cos(x[1]/2)
            sp = math.sin(x[1]/2)
            cr = math.cos(x[0]/2)
            sr = math.sin(x[0]/2)
            qw = cr * cp * cy + sr * sp * sy - 1
            qx = sr * cp * cy - cr * sp * sy
            qy = cr * sp * cy + sr * cp * sy
            qz = cr * cp * sy - sr * sp * cy
            q_int = qw**qw_deg * qx**qx_deg * qy**qy_deg * qz**qz_deg
            return q_int

        result, error = mcint.integrate(integrand, sampler(), measure=1, n=n_samples)
        return result

    # Minimize volume beneath the a(x).
    obj = a
    for i in non_q_idx:
        obj = obj.Integrate(z[i], z_min[i], z_max[i])
    cost = 0
    for monomial,coeff in obj.monomial_to_coefficient_map().items(): 
        qw_deg = monomial.degree(z[0]) 
        qx_deg = monomial.degree(z[1])
        qy_deg = monomial.degree(z[2]) 
        qz_deg = monomial.degree(z[3])
        monomial_int = integrate_quaternion_monte_carlo(qw_deg, qx_deg, qy_deg, qz_deg)
        cost += monomial_int * coeff
    poly = Polynomial(cost)
    cost_coeff = [c.Evaluate() for c in poly.monomial_to_coefficient_map().values()]
    # Make the numerics better
    cost = cost/np.max(np.abs(cost_coeff))
    cost = Polynomial(cost).RemoveTermsWithSmallCoefficients(1e-6)
    a_cost = prog.AddLinearCost(cost.ToExpression())

    # Enforce Bellman inequality.
    f_val = f(z, u_fixed)
    J_dot = J_expr.Jacobian(z).dot(f_val)

    LHS = a.ToExpression() - J_dot - l_cost(z, u_fixed)
    lam_deg = Polynomial(LHS).TotalDegree() - 2
    lam_deg = int(np.ceil(lam_deg/2)*2)
    # S procedure for qw^2 + qx^2  + qy^2 + qz^2= 1.
    lam = prog.NewFreePolynomial(Variables(z), lam_deg).ToExpression()
    S_ring = lam * ((z[0]+1)**2 + z[1]**2 + z[2]**2 + z[3]**2 - 1)
    if deg >= 0:
        S_Jdot = 0
        # Also constrain theta to be in [-pi/2, pi/2]
        for i in np.arange(nz):
            lam = prog.NewSosPolynomial(Variables(z), lam_deg)[0].ToExpression()
            S_Jdot += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
        if not actuator_saturate:
            prog.AddSosConstraint(LHS + S_ring + S_Jdot)
    else:
        prog.AddSosConstraint(LHS + S_ring)

    # Enforce that value function is PD
    if deg >= 0:
        S_J = 0
        # Also constrain theta to be in [-pi/2, pi/2]
        for i in np.arange(nz):
            lam = prog.NewSosPolynomial(Variables(z), deg-2)[0].ToExpression()
            S_J += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
        prog.AddSosConstraint(J_expr + S_J)
    else:
        prog.AddSosConstraint(J_expr)

    # Enforce l(x,u)-a(x) is PD
    u = prog.NewIndeterminates(nu, 'u')
    if deg >= 0:
        S_la = 0
        # Also constrain theta to be in [-pi/2, pi/2]
        for i in np.arange(nz):
            lam = prog.NewSosPolynomial(Variables(z), deg-2)[0].ToExpression()
            S_la += lam*(z[i]-z_max[i])*(z[i]-z_min[i])
        prog.AddSosConstraint(l_cost(z,u) - a.ToExpression() + S_la)
    else:
        prog.AddSosConstraint(l_cost(z,u) - a.ToExpression())

    # J(z0) = 0.
    J0 = J_expr.EvaluatePartial(dict(zip(z, z0)))
    prog.AddLinearConstraint(J0 == 0)

    # Actuator saturation
    if actuator_saturate:
        print("="*10, "Actuator saturation", "="*20)
        LHS_limits = []
        Su_limits = []
        S_ring_limits = []
        S_Jdot_limits = []   
        for k in range(3): #(-inf, u_min),[u_min, u_max], (u_max, inf) 
            print(k)
            u_limit = np.zeros(nu, dtype=Expression)
            Su_limit = 0
            if k == 0:
                u_limit[0] = u_min[0]
            elif k == 1:
                u_limit[0] = u_fixed[0]
            elif k == 2:
                u_limit[0] = u_max[0]
            for n in range(3):
                if n == 0:
                    u_limit[1] = u_min[1]
                elif n == 1:
                    u_limit[1] = u_fixed[1]
                elif n == 2:
                    u_limit[1] = u_max[1]
                for h in range(3):
                    if h == 0:
                        u_limit[2] = u_min[2]
                    elif h == 1:
                        u_limit[2] = u_fixed[2]
                    elif h == 2:
                        u_limit[2] = u_max[2]
                    for l in range(3):
                        if l == 0:
                            u_limit[3] = u_min[3]
                        elif l == 1:
                            u_limit[3] = u_fixed[3]
                        elif l == 2:
                            u_limit[3] = u_max[3]

                        f_limit = f(z, u_limit)
                        J_dot_limit = J_expr.Jacobian(z).dot(f_limit)
                        LHS_limit = a.ToExpression() - J_dot_limit - l_cost(z, u_limit)
                        LHS_limits.append(LHS_limit)

                        lam_u_deg = Polynomial(LHS_limit).TotalDegree() - np.max([Polynomial(u_limit[0]).TotalDegree(), Polynomial(u_limit[1]).TotalDegree()])
                        if k == 0:
                            lam_u = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                            Su_limit += lam_u*(u_fixed[0] - u_min[0])
                        elif k == 1:
                            lam_u_min = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                            lam_u_max = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                            Su_limit += lam_u_max*(u_fixed[0] - u_max[0]) + lam_u_min*(u_min[0] - u_fixed[0])
                        elif k == 2:
                            lam_u = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                            Su_limit += lam_u*(u_max[0] - u_fixed[0])

                        if n == 0:
                            lam_u = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                            Su_limit += lam_u*(u_fixed[1] - u_min[1])
                        elif n == 1:
                            lam_u_min = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                            lam_u_max = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                            Su_limit += lam_u_max*(u_fixed[1] - u_max[1]) + lam_u_min*(u_min[1] - u_fixed[1])
                        elif n == 2:
                            lam_u = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                            Su_limit += lam_u*(u_max[1] - u_fixed[1])

                        if h == 0:
                            lam_u = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                            Su_limit += lam_u*(u_fixed[2] - u_min[2])
                        elif h == 1:
                            lam_u_min = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                            lam_u_max = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                            Su_limit += lam_u_max*(u_fixed[2] - u_max[2]) + lam_u_min*(u_min[2] - u_fixed[2])
                        elif h == 2:
                            lam_u = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                            Su_limit += lam_u*(u_max[2] - u_fixed[2])

                        if l == 0:
                            lam_u = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                            Su_limit += lam_u*(u_fixed[3] - u_min[3])
                        elif l == 1:
                            lam_u_min = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                            lam_u_max = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                            Su_limit += lam_u_max*(u_fixed[3] - u_max[3]) + lam_u_min*(u_min[3] - u_fixed[3])
                        elif l == 2:
                            lam_u = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_u_deg/2)*2))[0].ToExpression()
                            Su_limit += lam_u*(u_max[3] - u_fixed[3])

                        lam_limit_deg = Polynomial(LHS_limit).TotalDegree() - 2
                        lam = prog.NewFreePolynomial(Variables(z), lam_limit_deg).ToExpression()
                        S_ring_limit = lam * ((z[0]+1)**2 + z[1]**2 + z[2]**2 + z[3]**2 - 1)
                        S_Jdot_limit = 0
                        for i in np.arange(nz):
                            lam = prog.NewSosPolynomial(Variables(z), lam_limit_deg)[0].ToExpression()
                            S_Jdot_limit += lam*(z[i]-z_max[i])*(z[i]-z_min[i])

                        Su_limits.append(Su_limit)
                        S_ring_limits.append(S_ring_limit)
                        S_Jdot_limits.append(S_Jdot_limit)
                        prog.AddSosConstraint(LHS_limit + S_ring_limit + S_Jdot_limit + Su_limit)


    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)
    assert result.is_success()
    a_star = result.GetSolution(a).RemoveTermsWithSmallCoefficients(1e-6).ToExpression()

    if actuator_saturate:
        LHS_limits_a_star = [result.GetSolution(x) for x in LHS_limits]

    prog.RemoveCost(a_cost)

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
            qw_deg = monomial.degree(z[0]) 
            qx_deg = monomial.degree(z[1])
            qy_deg = monomial.degree(z[2]) 
            qz_deg = monomial.degree(z[3])
            monomial_int = integrate_quaternion_monte_carlo(qw_deg, qx_deg, qy_deg, qz_deg)
            cost += monomial_int * coeff
        poly = Polynomial(cost)
        cost_coeff = [c.Evaluate() for c in poly.monomial_to_coefficient_map().values()]
        # Make the numerics better
        cost = cost/np.max(np.abs(cost_coeff))
        cost = Polynomial(cost).RemoveTermsWithSmallCoefficients(1e-6)
        prog.AddLinearCost(cost.ToExpression())

    # Enforce Bellman inequality.
    if actuator_saturate:
        for i in range(len(Su_limits)):
                prog.AddSosConstraint(LHS_limits_a_star[i] + S_ring_limits[i] + S_Jdot_limits[i] + Su_limits[i])
    else:
        prog.AddSosConstraint(a_star - J_dot - l_cost(z, u_fixed) + S_ring + S_Jdot)

    result = Solve(prog)
    assert result.is_success()
    J_star = Polynomial(result.GetSolution(J_expr)).RemoveTermsWithSmallCoefficients(1e-6)
    l_val = Polynomial(result.GetSolution(l_cost(z, u_fixed)))

    f2_val = f2(z)
    dJdz = J_star.ToExpression().Jacobian(z)
    u_star = - .5 * Rinv.dot(f2_val.T).dot(dJdz.T)

    if visualize:
        plot_value_function(J_star, u_star, z, z_max, z_min, u0, file_name="upper_bound_constrained_lqr_{}_{}".format(objective, deg), plot_states="xy", u_index=0)
    return J_star, z, z_max

if __name__ == '__main__':
    deg = 2
    J_star, z, z_max = quadrotor3d_sos_upper_bound(deg, objective="integrate_ring",visualize=True, actuator_saturate=True)

    C = extract_polynomial_coeff_dict(J_star, z)
    f = open("quadrotor3d/data/J_upper_bound_deg_{}.pkl".format(deg),"wb")
    pickle.dump(C, f)
    f.close()
