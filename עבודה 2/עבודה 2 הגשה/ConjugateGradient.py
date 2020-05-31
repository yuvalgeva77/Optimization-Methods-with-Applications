from numpy import linalg as LA
from numpy import matmul as mul

n = 256


def conjugateGradient(A, b, x, max_iteration=100, epsilon=10**(-10),w=1):
    r = b - mul(A, x)
    p = r
    iterations = [x]
    for k in range(max_iteration):
        A_p = mul(A, p)
        alpha = iner(r, r) / iner(p, A_p)
        iterations.append(iterations[k] + alpha * p)
        new_r = r - alpha * A_p
        if LA.norm(new_r) < epsilon:
            break
        beta = iner(new_r, new_r) / iner(r, r)
        p = new_r + beta * p
        r = new_r
    return iterations
