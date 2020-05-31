from numpy import linalg as LA
from numpy.linalg import inv
from numpy import matmul as mul
import numpy as np
from scipy.sparse import random
import scipy.sparse as sparse
import matplotlib.pyplot as plot


n = 256


def iner(a, b):
    return mul(a.transpose(), b)

# 1 a
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


# 1 b
def playQuestionOne():
    A = random(n, n, 5 / n, dtype=float)
    v = np.random.rand(n)
    v = sparse.spdiags(v, 0, v.shape[0], v.shape[0], 'csr')
    A = A.transpose() * v * A + 0.1*sparse.eye(n)
    x = np.zeros(n)
    b = np.random.rand(n)

    # ||Ax(k) -b||
    add_graph_line_first(jacobi, A.toarray(), b, x, 0.25, 'jacobi')
    add_graph_line_first(gaussSeidel, A.toarray(), b, x, 1, 'gauss Seidel')
    add_graph_line_first(steepestDescent, A.toarray(), b, x, 1, 'steepest Descent')
    add_graph_line_first(conjugateGradient, A.toarray(), b, x, 1, 'conjugate Gradient')
    plot.title("||Ax(k) -b||")
    plot.xlabel('Iterations')
    plot.ylabel('Residual Norm')
    plot.legend()
    plot.show()

    # ||Ax(k)−b|| / ||Ax(k−1)−b||
    add_graph_line_second(jacobi, A.toarray(), b, x, 0.25, 'jacobi')
    add_graph_line_second(gaussSeidel, A.toarray(), b, x, 1, 'gauss Seidel')
    add_graph_line_second(steepestDescent, A.toarray(), b, x, 1, 'steepest Descent')
    add_graph_line_second(conjugateGradient, A.toarray(), b, x, 1, 'conjugate Gradient')
    plot.title("||Ax(k)−b|| / ||Ax(k−1)−b||")
    plot.xlabel('Iterations')
    plot.ylabel('Residual Norm')
    plot.legend()
    plot.show()


def add_graph_line_first(fun, A, b, x, w, label, max_iterations=100):
    ans = fun(A, b, x, max_iterations, w=w)
    x_asix = list(map(lambda u: LA.norm(mul(A, u) - b), ans))
    plot.semilogy(x_asix, label=label)


def add_graph_line_second(fun, A, b, x, w, label, max_iterations=100):
    ans = fun(A, b, x, max_iterations, w=w)
    x_asix = []
    # ||Ax(k)−b|| / ||Ax(k−1)−b||
    for i in range(1,len(ans)):
        u = ans[i]
        prev_u = ans[i-1]
        norm = LA.norm(mul(A, u) - b) / LA.norm(mul(A, prev_u) -b)
        x_asix.append(norm)
    plot.semilogy(x_asix, label=label)



playQuestionOne()