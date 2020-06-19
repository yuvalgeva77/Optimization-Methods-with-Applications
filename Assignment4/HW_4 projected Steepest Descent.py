import numpy as np
from scipy.sparse import coo_matrix
import random
from numpy import linalg as LA
import matplotlib.pyplot as plt
from numpy import dot as mul


def pai(Y):
    ans=np.array([y[0] if y[0] >= 0 else 0 for y in Y])
    ans=np.reshape(ans, (ans.size, 1))
    return ans

def armijo(max_iter, x, f, gradient, direction, alpha=1, b=0.5,  c=10**-5):
    while max_iter > 0:
        objective = f(x + alpha * direction)
        limit = f(x) + c * alpha * mul(gradient.transpose(), direction)
        if objective <= limit:
            break
        alpha = alpha * b
        max_iter = max_iter - 1
    return alpha


def stop(x, next_x, epsilon):
    if LA.norm(x) != 0:
        return LA.norm(next_x - x) / LA.norm(x) < epsilon
    return False

#Projected Gradient Descent for LASSO regression
def desent(f, df, x_k, alpha= 0.01, epsilon = 10**-3,max_iter=100):
    x_k=pai(x_k)
    start_alpha = alpha
    output_fx=[]
    output_fx.append(f(x_k)[0])
    while max_iter > 0:
        gradient = df(x_k)
        # gradient_norm = LA.norm(gradient)
        normal_gradient = gradient
        direction = - normal_gradient

        alpha = armijo(10, x_k, f, normal_gradient, direction, start_alpha)
        next_x =pai( x_k + alpha * direction)

        if stop(x_k, next_x, epsilon):
            output_fx.append(f(next_x)[0])
            return (next_x,output_fx)

        x_k = next_x
        max_iter = max_iter - 1
        output_fx.append(f(x_k)[0])
    return (next_x,output_fx)



def f(A, b,l):
    n =int(len(A[1]))
    C1 = np.concatenate((np.eye(n), -1 * (np.eye(n))), axis=1)
    C2 = np.concatenate((np.eye(n),  (np.eye(n))), axis=1)
    one_t = np.ones((1, n))
    one_t_C2 = np.dot(one_t, C2)
    def fu(x):
        ACw_b = np.dot(np.dot(A, C1), x) - b
        ACw_b_norm = LA.norm(ACw_b, 'fro')
        return ACw_b_norm ** 2 + l * np.dot(one_t_C2,x)
    return fu

def derivative(A, b, l):
    n = int(len(A[1]))
    C1 = np.concatenate((np.eye(n), -1 * (np.eye(n))), axis=1)
    C2 = np.concatenate((np.eye(n), (np.eye(n))), axis=1)
    AC = np.dot(A, C1)
    ACT_Ac = np.dot(AC.transpose(), AC)
    one_t = np.ones((1, n))
    one_t_C2 = np.dot(one_t, C2)
    def grad(x):
        ACT_Ac_x= np.dot(ACT_Ac, x)
        ACT_Ac_x=np.reshape(ACT_Ac_x, (ACT_Ac_x.size, 1))
        return (2 * ACT_Ac_x - 2 * np.dot(AC.transpose(), b) + l * one_t_C2.transpose())
    return grad

#---------4c----------------------
def ass_c():
    A = np.random.normal(0, 1, (100, 200))
    row  = np.array(random.sample(range(200), k=20))
    col  = np.array([0 for _ in range(20)])
    data = np.array(np.random.normal(0, 1, (20)))
    x = coo_matrix((data, (row, col)), shape=(200, 1)).toarray()
    noice = np.random.normal(0, 0.1, (100, 1))
    b=np.dot(A,x)+noice
    lambdas =list(np.arange(0,20,0.7))


    for l in lambdas:
        (output_x, output_fx) = desent(f(A,b,l),derivative(A, b, l), np.zeros((400, 1)))
        print(output_fx)
        precent=np.count_nonzero(output_x)/len(output_x)*100
        if(precent<12 and 8<precent): #approximately 10% non-zero entries
            title="lambda:{0:,.1f},non zero:{1}%".format(l,precent)
            plt.plot([i for i in range (1,len(output_fx)+1)],output_fx,label=title)
    plt.xlabel("iteration")
    plt.ylabel("f(x)")
    plt.title("objective convergence")
    plt.legend()
    plt.show()
ass_c()