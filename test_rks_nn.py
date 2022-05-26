from cProfile import label
from cartpole.cartpole_rks_nn import setup_nn
import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

K = 10
h_layer = 16
sinks = []
for _ in range(K):
    sinks.append(setup_nn((2, h_layer, h_layer, 1), activation_type="sigmoid"))

dtype = torch.float64
X1, X2 = np.meshgrid(np.linspace(-1, 1, 101), np.linspace(-1, 1, 101))
X = np.vstack((X1.flatten(), X2.flatten()))
x_samples = torch.tensor(X.T, dtype=dtype)

basis = sinks[0](x_samples)
for k in range(1, K):
    b = sinks[k](x_samples)
    basis = torch.hstack((basis, b))

b = np.sum(X.T**2, axis=1)
A = basis.detach().numpy()
alpha = np.linalg.lstsq(A, b)[0]

fig = plt.figure(figsize=(9, 4))
ax = fig.subplots()
ax.set_xlabel("x")
ax.set_ylabel("theta")
ax.set_title("Cost-to-Go")
im = ax.imshow(b.reshape(X1.shape),
        cmap=cm.jet, aspect='auto',
        extent=(-1, 1, -1, 1))
ax.invert_yaxis()
fig.colorbar(im)
plt.savefig("original.png")


b_ = A@alpha
fig = plt.figure(figsize=(9, 4))
ax = fig.subplots()
ax.set_xlabel("x")
ax.set_ylabel("theta")
ax.set_title("Cost-to-Go")
im = ax.imshow(b_.reshape(X1.shape),
        cmap=cm.jet, aspect='auto',
        extent=(-1, 1, -1, 1))
ax.invert_yaxis()
fig.colorbar(im)
plt.savefig("rks_nn_fit.png")
# plt.show()



