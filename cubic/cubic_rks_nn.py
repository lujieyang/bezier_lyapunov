import time
from utils import *
from cubic_polynomial_fvi import optimal_cost_to_go
import torch
from torch.autograd import grad

from pydrake.all import (Solve, SolverOptions, CommonSolverOption, Polynomial, Variables, ge)
from scipy.integrate import quad
from pydrake.solvers import mathematicalprogram as mp

from scipy.special import comb

import matplotlib.pyplot as plt
from matplotlib import cm

def cubic_setup(dtype=torch.float64):
    nx = 1
    nu = 1

    def f1(x):
        return x - 4*x**3 
    
    def f2(x):
        return torch.ones([x.shape[0], nx, nu], dtype=dtype)

    # State limits (region of state space where we approximate the value function).
    x_max = torch.tensor([1], dtype=dtype)
    x_min = torch.tensor([-1], dtype=dtype)

    # Equilibrium point in both the system coordinates.
    x0 = torch.tensor([0], dtype=dtype)
        
    # Quadratic running cost in augmented state.
    Q = torch.diag(torch.tensor([1], dtype=dtype))  #*1e-3
    R = torch.diag(torch.tensor([1], dtype=dtype))  #*1e-3

    Rinv = torch.tensor(np.linalg.inv(R))
    params_dict = {"x_min": x_min, "x_max": x_max, "f1":f1, "Q":Q, "x0": x0, "f2": f2, "Rinv": Rinv, "nx": nx}
    return params_dict

def convex_sampling_hjb_lower_bound(K, params_dict, h_layer=16, activation_type="tanh", n_mesh=6, lam=1e-3, visualize=True):
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
        sinks.append(setup_nn((nx, h_layer, h_layer, 1), activation_type=activation_type))

    mesh_pts = []
    for i in range(nx):
        mesh_pts.append(torch.linspace(params_dict["x_min"][i], params_dict["x_max"][i], steps=n_mesh, dtype=dtype))
    x_samples = mesh_pts[0].unsqueeze(1)
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
        c = -l1[i].detach().numpy() *1e3
        b = -dPhi_dx_f1[i].detach().numpy().squeeze() *1e3
        Qin = 2* Q_batch[i].detach().numpy()/4 *1e3 # IMPORTANT: 0.5*x'*Q*x + b'*x + c <=0
        try:
            prog.AddQuadraticAsRotatedLorentzConeConstraint(
                Qin, b, c, alpha, psd_tol=1e-8)
        except:
            pass
    A = basis.detach().numpy()
    prog.AddLinearConstraint(A, np.zeros(A.shape[0]), np.inf*np.ones(A.shape[0]), alpha)
    # prog.AddLinearConstraint(A, eps*torch.sum(x_samples**2, dim=1).detach().numpy(), np.inf*np.ones(A.shape[0]), alpha)
    end_time = time.time()
    print("Time for adding constraints: ", end_time-start_time)

    integral = torch.sum(basis, dim=0)
    prog.AddLinearCost(-integral.detach().numpy(), alpha)
    # prog.Add2NormSquaredCost(lam*np.eye(K), np.zeros(K), alpha)

    # J(x0) = 0
    J0 = sinks[0](x0)
    for k in range(1, K):
        J0 = torch.hstack((J0, sinks[k](x0)))
    prog.AddLinearEqualityConstraint(J0.detach().numpy().dot(alpha) == 0)
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
        file_name="{}/h_layer_{}_mesh_{}_K_{}".format(activation_type, h_layer, n_mesh, K), check_inequality_gap=False)

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
    f1 = params_dict["f1"]
    f2 = params_dict["f2"]
    x0 = params_dict["x0"]
    Rinv = params_dict["Rinv"]
    Q = params_dict["Q"]
    dtype = x0.dtype

    X = torch.linspace(-1, 1, 101, dtype=dtype).unsqueeze(1)
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

    J = basis.detach().numpy() @ alpha_star
    x = X.detach().numpy()
    # plt.savefig("cubic/figures/rks_nn/{}.png".format(file_name))

    # U = - torch.einsum("ij, bjk->bik",Rinv, f2_dJdx)/2
    # U = U.squeeze().detach().numpy()
    # plt.plot(x, U)
    # plt.savefig("cubic/figures/rks_nn/{}_policy.png".format(file_name))

    RHS = (l1 - (quadratic/4).squeeze() + dJdx_f1.squeeze()).detach().numpy()
    if check_inequality_gap:
        plt.plot(x, RHS, label="K={}".format(K))
        # plt.savefig("cubic/figures/rks_nn/{}_inequality.png".format(file_name))
    else:
        plt.plot(x, J, label="K={}".format(K))

if __name__ == '__main__':
    n_mesh = 100
    h_layer = 8
    activation_type = "tanh"
    print("mesh: {}, activation: {}".format(n_mesh, activation_type))
    params_dict = cubic_setup()
    torch.random.manual_seed(88)
    for K in range(8, 20, 4):
        print("K = ", K)
        alpha, sinks = convex_sampling_hjb_lower_bound(K, params_dict, h_layer=h_layer, activation_type=activation_type, n_mesh=n_mesh, lam=5e-4, visualize=True)

    x_opt, J_opt = optimal_cost_to_go()
    plt.plot(x_opt, J_opt.T, 'k', label='J*')
    plt.legend()
    plt.savefig("cubic/figures/rks_nn/J_hjb_lower_bound.png")