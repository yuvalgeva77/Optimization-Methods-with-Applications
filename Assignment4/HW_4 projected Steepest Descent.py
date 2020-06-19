import numpy as np
from scipy.sparse import coo_matrix
import random
from numpy import linalg as ll
import matplotlib.pyplot as plt




def armijo(A, b, l, direction, w, maxSteps=100, beta=0.5, alpha0=1, c=1.0e-2):
    alpha = alpha0
    dir = direction(l, w)
    gr = (gradF(A,b,l, w)).transpose()
    gradTdir = np.dot(gr, dir) #np.dot(gardient(x).transpose(),d)
    for i in range(maxSteps):
        if calcF(A,b,l, w + alpha * dir) <= calcF(A,b,l, w) + c * alpha * gradTdir:
            return alpha
        else:
            alpha = beta * alpha
    return alpha


def  calcF(A, b, l, w):
    n =int(len(A[1]))
    C = np.concatenate((np.eye(n), -1 * (np.eye(n))), axis=1)
    ACw_b=np.dot(np.dot(A, C), w) - b
    ACw_b_norm=ll.norm(ACw_b, 'fro')
    F =ACw_b_norm**2 + l * np.dot(np.ones((1,len(w))) , w)
    return F

def gradF(A, b, l, w):
    n = int(len(A[1]))
    C = np.concatenate((np.eye(n), -1 * (np.eye(n))), axis=1)
    AC=np.dot(A, C)
    grad = 2* np.dot(np.dot(AC.transpose(), AC),w) - 2*np.dot(AC.transpose(), b) + l * np.ones((len(w),1))
    return grad


def getL(A, b):
    n = int(len(A[1]))
    C = np.concatenate((np.eye(n), -1 * (np.eye(n))), axis=1)
    lambdas =list(range(0,150,5))
    zeros = []
    for i in lambdas:
        [w, _] = gd(A, b, 400, i, 100)
        zeros.append(np.count_nonzero(w) / 200)

    plt.figure(1)
    plt.plot(lambdas, zeros)
    plt.xlabel("lambda")
    plt.ylabel("num of non-zero")
    plt.title("lambda(nnz)")
    plt.show()

    min_index = np.argmin(abs(np.array(zeros) - 0.1))
    l = lambdas[min_index]
    return l

def gd_loop(A, b, d, l, direction, maxSteps):
    # General line search method
    w = np.random.normal(0, 1, (d, 1))
    oldO = float('Inf')
    stopDelta = 10e-4
    objArr = []
    for i in range(maxSteps):
        o = calcF(A,b,l, w)
        objArr.append(o[0])
        if abs(o - oldO) < stopDelta:
            return (w,objArr)
        oldO = o
        alpha = armijo(A,b,l, direction, np.reshape(w,( w.size,1)), maxSteps)
        # alpha = 1
        lst = w + alpha * direction(l, w)
        w=np.array([x[0] if x[0] >= 0 else 0 for x in lst])
        print("A")
    return (w,objArr)

def gd(A,b,d, l, maxSteps=100):
    n = int(d/2)
    C = np.concatenate((np.eye(n), -1 * (np.eye(n))), axis=1)
    def getd(l,w):
       return - gradF(A,b,l,w)
    (w,objArr) = gd_loop(A,b,d, l, getd, maxSteps)
    w = np.dot(C,w)
    return (w,objArr)

A = np.random.normal(0, 1, (100, 200))
row  = np.array(random.sample(range(200), k=20))
col  = np.array([0 for _ in range(20)])
data = np.array(np.random.normal(0, 1, (20)))
x = coo_matrix((data, (row, col)), shape=(200, 1)).toarray()
noice = np.random.normal(0, 0.1, (100, 1))
b=np.dot(A,x)+noice
print(A)

l = getL(A, b)
(w, objArr) = gd(A, b, 400, l , 100)

plt.figure(2)
plt.plot([i for i in range (1,len(objArr)+1)], objArr)
plt.xlabel("iteration");
plt.ylabel("f(x)");
plt.title("objective convergence")
plt.show()
