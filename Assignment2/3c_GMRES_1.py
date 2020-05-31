import numpy as np
import matplotlib.pyplot as plt

#---------3.c ----------------------------------

def GMRES1(A, b, x_0, epsilon, max_iterations):
   res_list=[]
   r_k1=b-np.dot(A, x_0)
   x_k1=x_0
   for i in range(max_iterations):
       d=np.dot(A, r_k1)
       a=np.dot(np.transpose(r_k1),d)/np.dot(np.transpose(d),d)
       x_k=x_k1+a*r_k1
       r_k=r_k1-a*d
       residul_norm=np.linalg.norm(np.dot(A, x_k) - b)
       res_list.append(residul_norm)
       if (residul_norm < epsilon):
            plotGraph(res_list,range(i))
            return x_k
       x_k1 = x_k
       r_k1=r_k
   plotGraph(res_list,range(max_iterations))
   return x_k

def plotGraph(Xlist,Ylist):
    plt.semilogx(Xlist,Ylist)
    plt.ylabel("Iterations")
    plt.xlabel("Residul norm")
    plt.show()

A = np.array([[5.0, 4.0, 4.0,-1.0,0.0],[3.0, 12.0 ,4.0,-5.0,-5.0],[-4.0, 2.0, 6.0,0.0,3.0],[4.0, 5.0,-7.0,10.0,2.0],[1.0, 2.0, 5.0,3.0,10.0]])
b=np.array([[1.0],[1.0],[1.0],[1.0],[1.0]])
x_0=np.array([[0.0],[0.0],[0.0],[0.0],[0.0]])
max_iterations=50
epsilon=0
(GMRES1(A, b, x_0, epsilon, max_iterations))


