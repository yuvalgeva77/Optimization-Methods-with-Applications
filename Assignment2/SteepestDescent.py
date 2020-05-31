from numpy import linalg as LA
from numpy import matmul as mul

n = 256


def steepestDescent(A, b, x, max_iteration=100, epsilon=10**(-10),w=1):
    r = b - mul(A, x)
    iterations = [x]
    for k in range(max_iteration):
        A_r = mul(A, r)
        alpha = iner(r, r) / iner(r, A_r)
        iterations.append(iterations[k] + alpha * r)
        r = r - alpha * A_r
        norm = LA.norm(r)
        if norm <= epsilon:
            break
    return iterations
