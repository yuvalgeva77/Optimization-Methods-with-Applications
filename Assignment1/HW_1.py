from numpy import linalg as LA
import numpy as np


A = np.array([[1, 2, 3, 4], [2, 4, -4, 8], [-5, 4, 1, 5], [5, 0, -3, -7]])
At = A.transpose()
At_A = np.matmul(At, A)
w, v = LA.eig(At_A)
vector_index = np.argmax(w)
x = v[:, vector_index]
Ax = np.matmul(A, x)
norm = LA.norm(Ax, ord=2) / LA.norm(x, ord=2)
print("norm: ", norm)
print("vector x:", x)
