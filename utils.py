import numpy as np
from pydrake.all import (MathematicalProgram)

def monomial(x, n):
    return x**n


def calc_basis(z, coeff_shape, poly_func):
    Z = np.zeros(coeff_shape)
    it = np.nditer(Z, flags=['multi_index'])
    for x in it:
        idx = it.multi_index
        b = 1
        for dim in range(len(idx)):
            b *= poly_func(z[dim], idx[dim])
        Z[idx] = b
    return Z.flatten()


def calc_dJdz(z, coeff, poly_func):
    J = calc_value_function(z, coeff, poly_func)
    dJdz_expr = J.Jacobian(z)
    return dJdz_expr


def calc_value_function(x, J, poly_func):
    assert len(x) == len(J.shape)
    it = np.nditer(J, flags=['multi_index', 'refs_ok'])
    p = 0
    for k in it:
        b = np.copy(k)
        for dim, idx in enumerate(it.multi_index):
            b = b*poly_func(x[dim], idx)
        p += b
    return p


def calc_u_opt(dJdz, f2, Rinv):
    u_star = - .5 * Rinv.dot(f2.T).dot(dJdz.T)
    return u_star


def extract_polynomial_coeff_dict(poly, z):
    # return a dictionary of monomials and their coefficients
    nz = len(z)
    C = {}
    for monomial,coeff in poly.monomial_to_coefficient_map().items(): 
        m = []
        for i in range(nz):
            m.append(monomial.degree(z[i]))
        C[tuple(m)] = coeff.Evaluate({z[0]:0})
    return C


def reconstruct_polynomial_from_dict(C, z):
    poly = 0
    nz = len(z)
    for monomial in C.keys():
        m = 1
        for i in range(nz):
            m *= z[i]**monomial[i]
        poly += m * C[monomial]
    return poly
