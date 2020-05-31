from numpy import linalg as LA
from numpy.linalg import inv
from numpy import matmul as mul
import numpy as np

n = 256


def jacobi(A, b, x=np.zeros(n), max_iteration=1000, w=1, epsilon=10**(-10)):
    iterations = [x]
    D = np.diag(np.diag(A))
    inv_D = inv(D)

    for k in range(max_iteration):
        x = x + mul(w * inv_D, b - mul(A, x))
        iterations.append(x)
        r = mul(A, x) - b
        norm = LA.norm(r)
        if norm <= epsilon:
            break
    return iterations
