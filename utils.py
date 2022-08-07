import numpy as np
import pickle

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


def construct_monomial_basis_from_polynomial(J, nJ, z):
    nz = len(z)
    C = {}
    for m, coeff in J.monomial_to_coefficient_map().items():
        basis =[]
        for i in range(nz):
            basis.append(m.degree(z[i]))
        s = coeff.GetVariables().to_string()
        idx = int(s[s.find('(')+1:s.find(')')])
        c = coeff.Evaluate({list(coeff.GetVariables())[0]:1})
        C[idx] = (c, np.array(basis))

    def calc_basis(x):
        assert len(x.shape) == 2
        batch_size = x.shape[0]
        b = np.zeros([batch_size, nJ])
        for k in C.keys():
            c, power = C[k]
            b[:, k] = c * np.prod(x**power, axis=1)
        return b
    
    return calc_basis

def construct_polynomial_from_coeff(J, J_coeff):
    poly = 0
    for m, coeff in J.monomial_to_coefficient_map().items():
        s = coeff.GetVariables().to_string()
        idx = int(s[s.find('(')+1:s.find(')')])
        poly += m*J_coeff[idx]
    return poly

def save_polynomial(p, z, file_name):
    C = extract_polynomial_coeff_dict(p, z)
    f = open(file_name, "wb")
    pickle.dump(C, f)
    f.close()

def load_polynomial(z, file_name):
    with open(file_name, "rb") as input_file:
        C = pickle.load(input_file)
    p = reconstruct_polynomial_from_dict(C, z)
    return p

def matrix_adjoint(M):
    if len(M) == 2:
        return [[M[1, 1], -M[0, 1]],
                [-M[1, 0], M[0, 0]]]

    cofactors = []
    for r in range(len(M)):
        cofactorRow = []
        for c in range(len(M)):
            minor = matrix_minor(M, r, c)
            cofactorRow.append(((-1)**(r+c)) * matrix_det(minor))
        cofactors.append(cofactorRow)
    return np.array(cofactors).T

def matrix_det(M):
    if len(M) == 2:
        return M[0, 0] * M[1, 1]-M[0, 1] * M[1, 0]

    determinant = 0
    for c in range(len(M)):
        determinant += ((-1)**c)*M[0, c]*matrix_det(matrix_minor(M, 0, c))
    return determinant

def matrix_minor(M,i,j):
    return np.delete(np.delete(M, i, axis=0), j, axis=1)