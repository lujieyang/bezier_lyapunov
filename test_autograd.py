from torch.autograd import grad
from cartpole.cartpole_rks_nn import setup_nn
import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

K = 10
h_layer = 8
sinks = []
for _ in range(K):
    sinks.append(setup_nn((1, h_layer, h_layer, 1), activation_type="sigmoid"))

dtype = torch.float64
X = np.linspace(-np.pi, np.pi)
x_samples = torch.tensor(X.T, dtype=dtype).unsqueeze(1)
x_samples.requires_grad = True

basis = sinks[0](x_samples)
dPhi_dx = grad(basis, x_samples, grad_outputs=torch.ones_like(basis))[0].unsqueeze(2)
for k in range(1, K):
    b = sinks[k](x_samples)
    dphi_dx = grad(b, x_samples, grad_outputs=torch.ones_like(b))[0].unsqueeze(2)
    basis = torch.hstack((basis, b))
    dPhi_dx = torch.cat((dPhi_dx, dphi_dx), 2)

b = np.sin(X)
A = basis.detach().numpy()
alpha = np.linalg.lstsq(A, b)[0]
ddx = dPhi_dx.squeeze().detach().numpy()@alpha

plt.plot(X, np.cos(X), label="cos(x)")
plt.plot(X, ddx, label="rks_nn_fit_derivative")
plt.legend()
plt.savefig("test_autograd.png")





