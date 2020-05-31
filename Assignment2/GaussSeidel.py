from numpy import linalg as LA
from numpy.linalg import inv
from numpy import matmul as mul
import numpy as np

n = 256


def gaussSeidel(A, b, x, max_iteration=100,w=1, epsilon=10**(-10)):
    iterations = [x]

    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    l_and_d = inv(L + D)

    for k in range(max_iteration):
        x = iterations[k]
        x = x + mul( l_and_d, b - mul(A, x))
        iterations.append(x)
        r = mul(A, x) - b
        norm = LA.norm(r)
        if norm <= epsilon:
            break
    return iterations
