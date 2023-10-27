import numpy as np
import math

#Exercício 1:

d = np.array([[3.4, 0.5, 1.2, -0.8],
            [-1.6, 6.8, 1.1, 2.3],
            [0.1, -0.5, 2.5, 0.7],
            [-0.9, 1.7, 0.2, 4.9]]) 
d_t = np.transpose(d)  

a = d.dot(d_t)

#print("Matriz D:\n",d)
#print("Matriz D transposta:\n",d_t)
print("Exercício 1: \n Matriz A:\n",a)

#Exercício 2:

det_a = np.linalg.det(a) 
a_inv = np.linalg.inv(a)

print("Exercício 2: \n Determinante de A:",det_a, "\n Inversa da Matriz A:\n",a_inv)


#Exercício 3:

#v = np.linalg.eigvals(a)
v,w = np.linalg.eig(a)
#print("Valores Proprios de A:\n",v,"\n""Vetores proprios de A:\n",w)
print("Valores Proprios de A:\nv1:",v[0],"\nv2:",v[1],"\nv3:",v[2],"\nv4:",v[3])
print("Vetores Proprios de A:\nV1:",w[:,0],"\nV2:",w[:,1],"\nV3:",w[:,2],"\nV4:",w[:,3])








