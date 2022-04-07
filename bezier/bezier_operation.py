import numpy as np
from scipy.special import comb
import itertools
import matplotlib.pyplot as plt

from pydrake.all import (Variable, Variables, Polynomial)


def power_to_bernstein_poly(X):
    # The different interval other than [0,1] has impact on this function.
    # We deal with the general interval by first normalizing the variable to be 
    # in [0, 1], use check_poly_coeff_matrix to find the corresponding coefficients
    # in monomial basis.
    N = np.array(X.shape) - 1
    num_var = len(N)
    Z = np.zeros(num_var, dtype=int)  # multi-index with all 0's

    P = np.zeros(N + 1)
    S = construct_S(Z, N)
    for I in S:
        Js = construct_S(Z, I)
        PI = 0
        for J in Js:
            PI += bernstein_comb(I, J)/bernstein_comb(N, J) * X[J]
        P[I] = PI
    return P


def construct_S(L, U):
    # Construct the set S with all the combinations which are
    # greater than or equal to the multi-index L and smaller than 
    # or equal to U
    iterables = [range(t, k+1) for t, k in zip(L, U)]
    return itertools.product(*iterables)


def bernstein_comb(I, J):
    p = 1
    for k in range(len(I)):
        p *= comb(I[k], J[k])
    return p


def bernstein_add(F, G):
    Nf = np.array(F.shape) - 1
    Ng = np.array(G.shape) - 1
    if (Nf == Ng).all():
        H = F + G
    else:
        NE = np.maximum(Nf, Ng)
        F_ele = bernstein_degree_elevation(F, NE - Nf)
        G_ele = bernstein_degree_elevation(G, NE - Ng)
        H = F_ele + G_ele
    return H


def bernstein_degree_elevation(F, E):
    N = np.array(F.shape) - 1
    num_var = len(N)
    Z = np.zeros(num_var, dtype=int) # multi-index with all 0's
    dtype = type(F[tuple(Z)])
    H = np.zeros(N + E + 1, dtype=dtype)
    S = construct_S(Z, N + E)
    for K in S:
        D = np.maximum(Z, K - E)
        U = np.minimum(N, K)
        Ls = construct_S(D, U)
        HK = 0
        for L in Ls:
            K_L = tuple(np.array(K) - np.array(L))
            N_E = tuple(np.array(N) + np.array(E))
            HK += bernstein_comb(N, L) * bernstein_comb(E, K_L) / bernstein_comb(N_E, K) * F[L]
        H[K] = HK
    return H


def bernstein_mul(F, G, dtype=float):
    Nf = np.array(F.shape) - 1
    Ng = np.array(G.shape) - 1
    num_var = len(F.shape)
    N = Nf + Ng
    H = np.zeros(N+1, dtype=dtype)
    Z = np.zeros(num_var, dtype=int)  # multi-index with all 0's
    S = construct_S(Z, N)

    for K in S:
        D = np.maximum(Z, K - Ng)
        U = np.minimum(Nf, K)
        Ls = construct_S(D, U)
        HK = 0
        for L in Ls:
            K_L = tuple(np.array(K) - np.array(L))
            HK += bernstein_comb(Nf, L) * bernstein_comb(Ng, K_L) / bernstein_comb(N, K) * F[L] * G[K_L]
        H[K] = HK
    return H


def bernstein_derivative(X):
    N = np.array(X.shape) - 1
    num_var = len(N)
    Z = np.zeros(num_var, dtype=int)
    dtype = type(X[tuple(Z)])
    D = []

    for i in range(num_var):
        Ni = np.copy(N)
        Ni[i] -= 1
        Di = np.zeros(Ni + 1, dtype=dtype)
        S = construct_S(Z, N)
        for I in S:
            I_1 = np.copy(I)
            I_1[i] -= 1
            di = N[i]   # degree of i-th variable
            if (np.array(I) <= Ni).all():
                Di[I] += - di * X[I]
            if (I_1>=0).all():               
                I_1 = tuple(I_1)  # change np array into tuple for indexing               
                Di[I_1] += di * X[I]
        D.append(Di)
    return D


def bernstein_integral(X, lo=0, up=1):
    end = bernstein_definite_integral(X, up)
    start = bernstein_definite_integral(X, lo)
    return end-start


def bernstein_definite_integral(X, val):
    dim = np.array(X.shape)
    if val == 0:
        return 0
    elif val == 1:
        return np.sum(X)/np.product(dim)
    else:
        it = np.nditer(X, flags=['multi_index', 'refs_ok'])
        integral = 0
        for x in it:
            idx = it.multi_index
            int_idx = 1
            for d in range(len(idx)):
                n = dim[d] - 1
                k = idx[d]
                b = 0
                for j in range(k+1, n+2):
                    b += BernsteinPolynomial(val, j, n+1)
                int_idx *= b/(n+1)
            integral += X[idx] * int_idx
        return integral


def BezierCurve(t, x):
    if len(x) == 1:
        return x[0]
    return (1-t)*BezierCurve(t, x[:-1]) + t*BezierCurve(t, x[1:])


def BernsteinPolynomial(t, i, n, lo=0, up=1):
    c = comb(n, i)
    return c * (t-lo)**i * (up-t)**(n-i) / (up-lo)**n


# a multi-dimensional Bezier surface in the variables x with degrees K.shape-1
# (and coefficients K).
def BezierSurface(x, K):
    assert len(x) == len(K.shape)
    it = np.nditer(K, flags=['multi_index', 'refs_ok'])
    p = 0
    for k in it:
        b = np.copy(k)
        for dim, idx in enumerate(it.multi_index):
            b *= BernsteinPolynomial(x[dim], idx, K.shape[dim]-1)
        p += b
    return p


def check_poly_coeff_matrix(f):
    # f = lambda x, u: x - 4 * x ** 3 - u
    # l = lambda x, u: x ** 2 + u ** 2

    x = Variable('x')
    u = Variable('u')
    xu = Variables([x, u])

    print(Polynomial(f(x, u), xu).monomial_to_coefficient_map())


def check_coeff_positivity():
    # f(x) = (x-0.2)^2
    f = np.array([0.04, -0.4, 1])

    f_bern = power_to_bernstein_poly(f)
    deg = len(f_bern) - 1
    while (f_bern < 0).any():
        f_bern = bernstein_degree_elevation(f_bern, np.array([1]))
        deg += 1
        if deg % 10 == 0:
            print("Degree: {}, min coeff: {}".format(deg, np.min(f_bern)))
            # plot_bezier(f_bern, -1, 1)
    print(deg)


def verify_half_domain():
    # f(x) = x^2
    f = np.array([0, 0, 1])

    f_bern = power_to_bernstein_poly(f)
    plot_bezier(f_bern, -1, 1)


def plot_bezier(f_bern, x_lo, x_up, label="f(x)"):
    deg = len(f_bern) - 1

    n_breaks = 101
    x = np.linspace(x_lo, x_up, n_breaks)
    y = np.zeros(n_breaks)

    for d in range(deg + 1):
        y += f_bern[d] * BernsteinPolynomial(x, d, deg)

    plt.plot(x, y, label=label)
    # plt.show()


def plot_energy(V):
    n_points = 51
    theta = np.linspace(-np.pi, np.pi, n_points)
    thetadot = np.linspace(-10, 10, n_points)

    E = np.zeros([n_points, n_points])
    for i in range(n_points):
        t = theta[i]
        for j in range(n_points):
            td = thetadot[j]
            E[i, j] = BezierSurface(np.array([np.sin(t), np.cos(t), td]), V)
    [X, Y] = np.meshgrid(theta, thetadot)
    plt.contourf(X, Y, E)
    plt.colorbar()
    plt.savefig("Energy.png")
