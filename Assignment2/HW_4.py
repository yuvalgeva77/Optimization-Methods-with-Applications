import numpy as np
from numpy.linalg import inv
from numpy import matmul as mul
from numpy import linalg as LA
import matplotlib.pyplot as plot


def iner(a, b):
    return mul(a.transpose(), b)


# define L
array_l = '2 -1 -1 0 0 0 0 0 0 0 \
-1 2 -1 0 0 0 0 0 0 0 \
-1 -1 3 -1 0 0 0 0 0 0 \
0 0 -1 5 -1 0 -1 0 -1 -1 \
0 0 0 -1 4 -1 -1 -1 0 0 \
0 0 0 0 -1 3 -1 -1 0 0 \
0 0 0 -1 -1 -1 5 -1 0 -1 \
0 0 0 0 -1 -1 -1 4 0 -1 \
0 0 0 -1 0 0 0 0 2 -1 \
0 0 0 -1 0 0 -1 -1 -1 4'

all_numbers = list(map(int, array_l.split(' ')))
L = np.array([all_numbers])
L = L.reshape(10, 10)

v = [1]*10
# define b
b = v
for i in range(10):
    if i % 2 == 1:
        b[i] *= -1


# def ass_a():
#     x = np.zeros(10)
#     x_asix = list(map(lambda u: LA.norm(mul(L, u) - b), jacobi(L, b, x,  epsilon=1e-5)))
#     print("iteration number of 4a:", len(x_asix))
#     plot.semilogy(x_asix, label='jacobi')
#     plot.title("question 4a")
#     plot.show()


def ass_a():
    x = np.zeros(10)

    iterations = [x]
    D = np.diag(np.diag(L))
    inv_D = inv(D)
    norm_list = []
    for k in range(100):
        x = x + mul(inv_D, b - mul(L, x))
        iterations.append(x)
        r = mul(L, x) - b
        norm = LA.norm(r)
        norm_list.append(norm)
        if norm <= 10**-5:
            break

    print("iteration number of 4a:", k)
    plot.semilogy(norm_list, label='jacobi')
    plot.title("question 4a")
    plot.show()

def ass_b():
    m1 = L[:3, :3]
    m2 = L[3:, 3:]
    inv_m1 = inv(m1)
    inv_m2 = inv(m2)
    inv_m = np.zeros(100).reshape(10, 10)
    inv_m[:3, :3] = inv_m1
    inv_m[3:, 3:] = inv_m2
    x = np.zeros(10)

    iterations = [x]
    for k in range(100):
        x = x + mul(0.7 * inv_m, b - mul(L, x))
        iterations.append(x)
        r = mul(L, x) - b
        norm = LA.norm(r)
        if norm <= 10**-5:
            break

    print("iteration number of 4b:", k)
    x_asix = list(map(lambda u: LA.norm(mul(L, u) - b), iterations))
    plot.semilogy(x_asix, label='jacobi')
    plot.title("question 4b")
    plot.show()


def ass_c():
    m1 = L[:4, :4]
    m2 = L[4:7, 4:7]
    m3 = L[7:, 7:]
    inv_m1 = inv(m1)
    inv_m2 = inv(m2)
    inv_m3 = inv(m3)

    inv_m = np.zeros(100).reshape(10, 10)
    inv_m[:4, :4] = inv_m1
    inv_m[4:7, 4:7] = inv_m2
    inv_m[7:, 7:] = inv_m3

    x = np.zeros(10)

    iterations = [x]
    for k in range(100):
        x = x + mul(0.7 * inv_m, b - mul(L, x))
        iterations.append(x)
        r = mul(L, x) - b
        norm = LA.norm(r)
        if norm <= 10**-5:
            break

    print("iteration number of 4c:", k)
    x_asix = list(map(lambda u: LA.norm(mul(L, u) - b), iterations))
    plot.semilogy(x_asix, label='jacobi')
    plot.title("question 4c")
    plot.show()


ass_a()
ass_b()
ass_c()
