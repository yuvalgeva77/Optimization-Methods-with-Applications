from numpy import linalg as LA
import numpy as np
from numpy import matmul as mul

# ----------------------------------------A-------------------------------


def gramSchmidtQR(a):
    m, n = a.shape
    # initiation
    r = np.zeros(shape=(n, n))
    q = np.zeros(shape=(m, n))
    a1 = a[:, 0]
    r[0][0] = LA.norm(a1, ord=2)
    q[:, 0] = a1 / r[0][0]

    for i in range(1, n):
        ai = a[:, i]
        q[:, i] = ai
        for j in range(0, i):
            qj = q[:, j]
            r[j][i] = mul(qj.transpose(), ai)
            q[:, i] = q[:, i] - r[j][i]*qj
        r[i][i] = LA.norm(q[:, i], ord=2)
        q[:, i] = q[:, i]/r[i][i]
    return q, r


def modifiedGramSchmidtQR(a):
    m, n = a.shape
    # initiation
    r = np.zeros(shape=(n, n))
    q = np.zeros(shape=(m, n))
    a1 = a[:, 0]
    r[0][0] = LA.norm(a1, ord=2)
    q[:, 0] = a1 / r[0][0]

    for i in range(1, n):
        ai = a[:, i]
        q[:, i] = ai
        for j in range(0, i):
            qj = q[:, j]
            qi = q[:, i]
            r[j][i] = mul(qj.transpose(), qi)
            q[:, i] = q[:, i] - r[j][i]*qj
        r[i][i] = LA.norm(q[:, i], ord=2)
        q[:, i] = q[:, i]/r[i][i]
    return q, r

# ----------------------------------------B-------------------------------


def get_matrix(epsilon):
    return np.array([[1, 1, 1], [epsilon, 0, 0], [0, epsilon, 0], [0, 0, epsilon]])


def print_Q_R(method, epsilon, q, r):
    A = get_matrix(epsilon)
    print(method,", epsilon =", epsilon)
    print("Q:\n", q)
    print("R:\n", r)


#   epsilon = 1
A = get_matrix(1)
GS_Q_1, GS_R_1 = gramSchmidtQR(A)
modified_GS_Q_1, modified_GS_R_1 = modifiedGramSchmidtQR(A)


# epsilon = 1e-10
A = get_matrix(1e-10)
GS_Q_e, GS_R_e = gramSchmidtQR(A)
modified_GS_Q_e, modified_GS_R_e = modifiedGramSchmidtQR(A)


def b_print_all_fractation():
    print_Q_R("Gram Schmidt", 1, GS_Q_1, GS_R_1)
    print_Q_R("Modified Gram Schmidt", 1, modified_GS_Q_1, modified_GS_R_1)
    print_Q_R("Gram Schmidt", 1e-10, GS_Q_e, GS_R_e)
    print_Q_R("Modified Gram Schmidt", 1e-10, modified_GS_Q_e, modified_GS_R_e)


# ----------------------------------------B-------------------------------
def c():

    I = np.identity(4)
    frobenius = list()
    # 1
    Qt_Q_1 = mul(GS_Q_1, GS_Q_1.transpose())
    frobenius.append(LA.norm(Qt_Q_1 ))
    print(frobenius[-1])

    # 2
    modified_Qt_Q_1 = mul(modified_GS_Q_1, modified_GS_Q_1.transpose())
    frobenius.append(LA.norm(modified_Qt_Q_1))
    print(frobenius[-1])

    # 3
    Qt_Q_e = mul(GS_Q_e, GS_Q_e.transpose())
    frobenius.append(LA.norm(Qt_Q_e ))
    print(frobenius[-1])

    # 4
    modified_Qt_Q_e = mul(modified_GS_Q_e, modified_GS_Q_e.transpose())
    frobenius.append(LA.norm(modified_Qt_Q_e))
    print(frobenius[-1])

    print("min: ", min(frobenius)," index: ",frobenius.index(min(frobenius)))

