from numpy import matmul as mul
from numpy.linalg import inv
import numpy as np
from functools import reduce

A = np.array([[2,1,2], [1,-2,1], [1,2,3], [1,1,1]])
b = np.array([[6], [1], [5], [2]])


# Cholesky factorization
def cholesky_factorization():
    At = A.transpose()
    At_A = mul(At, A)
    L = np.linalg.cholesky(At_A)
    Lt = L.transpose()
    x = reduce(mul, [inv(Lt), inv(L), At, b])
    r = mul(A,x) - b
    print("X least squares via Cholesky factorization:\n", x)
    print("r:\n", r)


# QR
def qr():
    Q, R = np.linalg.qr(A)
    x_qr = reduce(mul, [inv(R), Q.transpose(), b])
    r = mul(A,x_qr) - b
    print("least squares via QR factorization:\n",x_qr)
    print("r:\n", r)


# SVD
def svd():
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    s = np.diag(s)
    x_svd = reduce(mul, [vt.transpose(), inv(s), u.transpose(), b])
    r = mul(A,x_svd) - b
    print("least squares via SVD\n",x_svd)
    print("r:\n", r)


# weighted least squares
def weighted_least_squares():
    At = A.transpose()
    w = np.array([1000,1, 1, 1])
    w = np.diag(w)
    At_w_A = reduce(mul, [At, w, A])
    x_weighted = reduce(mul, [inv(At_w_A), At, w, b])
    r = mul(A, x_weighted) - b
    print("X weighted least squares\n", x_weighted)
    print(r)
    print(r[0][0])
    print(abs(r[0][0]) < 1/1000 )