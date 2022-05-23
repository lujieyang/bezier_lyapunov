import time
from utils import *
import torch
from torch.autograd import grad

from pydrake.all import (Solve, SolverOptions, CommonSolverOption, Polynomial, Variables, ge)
from scipy.integrate import quad
from pydrake.solvers import mathematicalprogram as mp
import pydrake.symbolic as sym
from pydrake.symbolic import Expression

from scipy.special import comb

import matplotlib.pyplot as plt
from matplotlib import cm

def cartpole_setup(dtype=torch.float64):
    nz = 5
    nq = 2
    nx = 2 * nq
    nu = 1

    mc = 10
    mp = 1
    l = .5
    g = 9.81

    def f1(x):
        s = torch.sin(x[:, 1])
        c = torch.cos(x[:, 1])
        qdot = x[:, nq:]
        f1_val = torch.zeros(x.shape[0], nx, dtype=dtype)
        f1_val[:, :nq] = qdot
        f1_val[:, 2] = (mp*s*(l*qdot[:, 1]**2+g*c))/(mc+mp*s**2)
        f1_val[:, 3] = (- mp*l*qdot[:, 1]**2*c*s - (mc+mp)*g*s)/(mc+mp*s**2)/l
        return f1_val 
    
    def f2(x):
        s = torch.sin(x[:, 1])
        c = torch.cos(x[:, 1])
        f2_val = torch.zeros([x.shape[0], nx, nu], dtype=dtype)
        f2_val[:, 2, :] = (1/(mc+mp*s**2)).unsqueeze(1)
        f2_val[:, 3, :] = (-c/(mc+mp*s**2)/l).unsqueeze(1)
        return f2_val

    # State limits (region of state space where we approximate the value function).
    x_max = torch.tensor([2, 2*np.pi, 6, 6], dtype=dtype)
    x_min = torch.tensor([-2, 0, -6, -6], dtype=dtype)

    # Equilibrium point in both the system coordinates.
    x0 = torch.tensor([0, np.pi, 0, 0], dtype=dtype)
        
    # Quadratic running cost in augmented state.
    Q = torch.diag(torch.tensor([10, 10, 1, 1], dtype=dtype))*1e-3
    R = torch.diag(torch.tensor([1], dtype=dtype))*1e-3

    Rinv = torch.tensor(np.linalg.inv(R))
    params_dict = {"x_min": x_min, "x_max": x_max, "f1":f1, "Q":Q, "x0": x0,
                   "nz": nz, "f2": f2, "Rinv": Rinv, "nq": nq, "nx": nx}
    return params_dict

def convex_sampling_hjb_lower_bound(K, params_dict, h_layer=32, n_mesh=6, eps=1e-5, visualize=True):
    # Sample for nonnegativity constraint of HJB RHS
    nx = params_dict["nx"]
    f1 = params_dict["f1"]
    f2 = params_dict["f2"]
    Rinv = params_dict["Rinv"]
    Q = params_dict["Q"]
    x0 = params_dict["x0"]
    dtype = x0.dtype

    prog = MathematicalProgram()
    alpha = prog.NewContinuousVariables(K, "alpha")
    sinks = []
    for _ in range(K):
        sinks.append(setup_nn((nx, h_layer, h_layer, 1)))

    mesh_pts = []
    for i in range(nx):
        mesh_pts.append(torch.linspace(params_dict["x_min"][i], params_dict["x_max"][i], steps=n_mesh, dtype=dtype))
    mesh_pts = torch.meshgrid(mesh_pts[0], mesh_pts[1], mesh_pts[2], mesh_pts[3])

    x_samples = mesh_pts[0].flatten()
    for i in range(1, nx):
        x_samples = torch.vstack((x_samples, mesh_pts[i].flatten()))
    x_samples = x_samples.T
    x_samples.requires_grad = True

    basis = sinks[0](x_samples)
    dPhi_dx = grad(basis, x_samples, grad_outputs=torch.ones_like(basis))[0].unsqueeze(2)
    for k in range(1, K):
        b = sinks[k](x_samples)
        dphi_dx = grad(b, x_samples, grad_outputs=torch.ones_like(b))[0].unsqueeze(2)
        basis = torch.hstack((basis, b))
        dPhi_dx = torch.cat((dPhi_dx, dphi_dx), 2)
    
    f2_val = f2(x_samples)
    f2_dPhi_dx = torch.bmm(torch.transpose(f2_val, 1, 2), dPhi_dx)
    Q_batch = torch.einsum("bki, ii, bik->bik", f2_dPhi_dx, Rinv, f2_dPhi_dx)

    l1 = torch.einsum("bi, ij, bj->b", x_samples-x0, Q, x_samples-x0)
    f1_val = f1(x_samples)
    dPhi_dx_f1 = torch.bmm(torch.transpose(dPhi_dx, 1, 2), f1_val.unsqueeze(2))

    start_time = time.time()
    for i in range(len(l1)):
        if i % 1000 == 0:
            print("Mesh x0 No.", i)
        c = -l1[i].detach().numpy() 
        b = -dPhi_dx_f1[i].detach().numpy() 
        Qin = Q_batch[i].detach().numpy()/4
        try:
            prog.AddQuadraticAsRotatedLorentzConeConstraint(
                Qin, b, c, alpha, psd_tol=1e-5)
        except:
            pass
    A = 100 * basis.detach().numpy()
    # prog.AddLinearConstraint(A, np.zeros(A.shape[0]), np.inf*np.ones(A.shape[0]), alpha)
    prog.AddLinearConstraint(A, eps*torch.sum(x_samples**2, dim=1).detach().numpy(), np.inf*np.ones(A.shape[0]), alpha)
    end_time = time.time()
    print("Time for adding constraints: ", end_time-start_time)

    integral = torch.trapz(basis, dim=0)
    prog.AddLinearCost(-integral.detach().numpy(), alpha)

    # J(x0) = 0
    J0 = sinks[0](x0)
    for k in range(1, K):
        J0 = torch.hstack((J0, sinks[k](x0)))
    prog.AddLinearEqualityConstraint(J0.detach().numpy().reshape(1, K), np.zeros([1, 1]), alpha)
    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)

    print("="*10, "Solving","="*20)
    solve_start = time.time()
    result = Solve(prog)
    solve_end = time.time()
    print("Time for solving: ", solve_end-solve_start)

    alpha_star = result.GetSolution(alpha)

    assert result.is_success()
    print("area: ", -result.get_optimal_cost())
    print("# of quadratic constraints: ", len(
        prog.rotated_lorentz_cone_constraints()))

    if visualize:
        plot_value_function(alpha_star, params_dict, sinks, K,
        file_name="h_layer_{}_mesh_{}_K_{}".format(h_layer, n_mesh, K))

    return alpha_star, sinks

def setup_nn(nn_layer_width: tuple,
               bias: bool = True,
               activation_type="relu",
               dtype=torch.float64):
    """
    Setup a nn network.
    @param bias whether the linear layer has bias or not.
    """
    assert (isinstance(nn_layer_width, tuple))

    if activation_type == "relu":
        activation = torch.nn.ReLU()
    elif activation_type == "tanh":
        activation = torch.nn.Tanh()
    elif activation_type == "sigmoid":
        activation = torch.nn.Sigmoid()

    linear_layers = [None] * (len(nn_layer_width) - 1)

    for i in range(len(linear_layers)):
        next_layer_width = nn_layer_width[i + 1]
        linear_layers[i] = torch.nn.Linear(nn_layer_width[i],
                                           next_layer_width,
                                           bias=bias).type(dtype)

    layers = [None] * (len(linear_layers) * 2 - 1)
    for i in range(len(linear_layers) - 1):
        layers[2 * i] = linear_layers[i]
        layers[2 * i + 1] = activation
    layers[-1] = linear_layers[-1]
    nn = torch.nn.Sequential(*layers)
    return nn

def plot_value_function(alpha_star, params_dict, sinks, K, file_name="", check_inequality_gap=True):
    x_min = params_dict["x_min"]
    x_max = params_dict["x_max"]
    f1 = params_dict["f1"]
    f2 = params_dict["f2"]
    x0 = params_dict["x0"]
    Rinv = params_dict["Rinv"]
    Q = params_dict["Q"]
    dtype = x0.dtype

    X1, X2 = torch.meshgrid(torch.linspace(x_min[0], x_max[0], 51, dtype=dtype),
                    torch.linspace(x_min[1], x_max[1], 51, dtype=dtype))
    # X = torch.vstack((X1.flatten(), X2.flatten(), torch.random.random(51*51), torch.random.random(51*51)))
    X = torch.vstack((X1.flatten(), X2.flatten(), torch.zeros(51*51, dtype=dtype), torch.zeros(51*51, dtype=dtype))).T
    X.requires_grad = True

    basis = sinks[0](X)
    dPhi_dx = grad(basis, X, grad_outputs=torch.ones_like(basis))[0].unsqueeze(2)
    for k in range(1, K):
        b = sinks[k](X)
        dphi_dx = grad(b, X, grad_outputs=torch.ones_like(b))[0].unsqueeze(2)
        basis = torch.hstack((basis, b))
        dPhi_dx = torch.cat((dPhi_dx, dphi_dx), 2)
    
    dJdx = (dPhi_dx @ alpha_star).unsqueeze(2)
    f2_val = f2(X)
    f2_dJdx = torch.bmm(torch.transpose(f2_val, 1, 2), dJdx)
    quadratic = torch.einsum("bki, ii, bik->bik", f2_dJdx, Rinv, f2_dJdx)

    l1 = torch.einsum("bi, ij, bj->b", X-x0, Q, X-x0)
    f1_val = f1(X)
    dJdx_f1 = torch.bmm(torch.transpose(dJdx, 1, 2), f1_val.unsqueeze(2))

    J = basis.detach().numpy() @ alpha_star * 1e3
    fig = plt.figure(figsize=(9, 4))
    ax = fig.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("theta")
    ax.set_title("Cost-to-Go")
    im = ax.imshow(J.reshape(X1.shape),
            cmap=cm.jet, aspect='auto',
            extent=(x_min[0], x_max[0], x_max[1], x_min[1]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.savefig("cartpole/figures/rks_nn/{}.png".format(file_name))

    U = - torch.einsum("ij, bjk->bik",Rinv, f2_dJdx)/2
    U = U.squeeze().detach().numpy()
    fig = plt.figure(figsize=(9, 4))
    ax = fig.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("theta")
    ax.set_title("Policy")
    im = ax.imshow(U.reshape(X1.shape),
            cmap=cm.jet, aspect='auto',
            extent=(x_min[0], x_max[0], x_max[1], x_min[1]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.savefig("cartpole/figures/rks_nn/{}_policy.png".format(file_name))

    RHS = (l1 - (quadratic/4).squeeze() + dJdx_f1.squeeze()).detach().numpy()
    if check_inequality_gap:
        fig = plt.figure(figsize=(9, 4))
        ax = fig.subplots()
        ax.set_xlabel("x")
        ax.set_ylabel("theta")
        ax.set_title("Bellman Inequality")
        im = ax.imshow(RHS.reshape(X1.shape),
                cmap=cm.jet, aspect='auto',
                extent=(x_min[0], x_max[0], x_max[1], x_min[1]))
        ax.invert_yaxis()
        fig.colorbar(im)
        plt.savefig("cartpole/figures/rks_nn/{}_inequality.png".format(file_name))

if __name__ == '__main__':
    K = 20
    n_mesh = 10
    h_layer = 64
    params_dict = cartpole_setup()
    torch.random.manual_seed(88)
    alpha, sinks = convex_sampling_hjb_lower_bound(K, params_dict, h_layer=h_layer, n_mesh=n_mesh, visualize=True)

    sink_dict = {"alpha": alpha}
    for k in range(K):
        sink_dict[k] = sinks[k]
    torch.save(sink_dict, "cartpole/data/rks_nn/alpha_{}_sink_{}_mesh_{}_hidden.pth".format(K, n_mesh, h_layer))