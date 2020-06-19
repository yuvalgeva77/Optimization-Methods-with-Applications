# Write a function that, given data matrix X and labels, computes the logistic
# regression objective, its gradient, and its Hessian matrix (assume that the Hessian
# matrix is not so big).
import numpy as np
from scipy.sparse import coo_matrix
import random
from numpy import linalg as ll
import matplotlib.pyplot as plt

def pai(Y):
    return np.array([y[0] if y[0] >= 0 else 0 for y in Y])


def armijo(x,d, f,  gardient, b=0.5, a_0=1, c=1,max_iterations=100):
    a_k=a_0
    for i in range(0,max_iterations):
        f_k = f(x+ a_k * d)
        limit = f(x) + c * a_k * np.dot(gardient(x).transpose(),d)
        if f_k <= limit :
           return a_k
        a_k = a_k * b
    return a_k


def gradient_descent(f, gardient, x_0, alpha= 0.01, eps = 10**-3,max_iterations=100):
    x_k=np.clip(x_0, -1, 1)
    output_x=[]
    output_fx = []
    for i in range(0,max_iterations):
        d_sd = -gardient(x_k)
        # d_sd = np.reshape(d_sd, d_sd.size)
        # np.reshape(w, (w.size, 1))
        # a = armijo(x_k, -d_sd, f, gardient)
        a=1
        next_x = pai(x_k +a*d_sd)
        if(np.linalg.norm(x_k)!=0):
            if np.linalg.norm(next_x - x_k)/np.linalg.norm(x_k) < eps:
                output_x = output_x + [next_x]
                output_fx.append(f(next_x)[0])
                return (output_x,output_fx)
        # output_x = output_x.append(next_x)
        output_fx.append(f(next_x)[0])
        x_k = next_x
    return (output_x, output_fx)



def f(A, b,l):
    n =int(len(A[1]))
    C = np.concatenate((np.eye(n), -1 * (np.eye(n))), axis=1)
    def fu(x):
       ACw_b=np.dot(np.dot(A, C), x) - b
       ACw_b_norm=ll.norm(ACw_b, 'fro')
       return ACw_b_norm**2 + l * np.dot(np.ones((1,len(x))) , x)
    return fu

def gradient(A, b, l):
    n = int(len(A[1]))
    C = np.concatenate((np.eye(n), -1 * (np.eye(n))), axis=1)
    AC = np.dot(A, C)
    def gard(x):
        return 2 * np.dot(np.dot(AC.transpose(), AC), x) - 2 * np.dot(AC.transpose(), b) + l * np.ones((len(x), 1))
    return gard

# -----------------------------------------------------------



A = np.random.normal(0, 1, (100, 200))
row  = np.array(random.sample(range(200), k=20))
col  = np.array([0 for _ in range(20)])
data = np.array(np.random.normal(0, 1, (20)))
x = coo_matrix((data, (row, col)), shape=(200, 1)).toarray()
noice = np.random.normal(0, 0.1, (100, 1))
b=np.dot(A,x)+noice
lambdas =list(range(0,150,5))
for l in lambdas:
    (output_x, output_fx) = gradient_descent(f(A,b,l),gradient(A, b, l), np.random.normal(0, 1, (400, 1)))
    if(np.count_nonzero(output_x) / 50): #approximately 10% non-zero entries
        plt.plot([i for i in range (1,len(output_fx)+1)],output_fx)
        plt.xlabel("iteration");
        plt.ylabel("f(x)");
        plt.title("objective convergence")
        plt.show()