import numpy as np
import math
import scipy as sp

#Exercício 3

#definir função que devolve o vetor correspondente de quaterniões
    #entrada em graus saída em radianos
def eta_v0(x):
    a = np.radians(x[0])
    b = np.radians(x[1])
    c = np.radians(x[2])
    
    eta_0 = np.cos(a/2)*np.cos(b/2)*np.cos(c/2)+np.sin(a/2)*np.sin(b/2)*np.sin(c/2)
    eta_1 = np.sin(a/2)*np.cos(b/2)*np.cos(c/2)-np.cos(a/2)*np.sin(b/2)*np.sin(c/2)
    eta_2 = np.cos(a/2)*np.sin(b/2)*np.cos(c/2)+np.sin(a/2)*np.cos(b/2)*np.sin(c/2)
    eta_3 = np.cos(a/2)*np.cos(b/2)*np.sin(c/2)-np.sin(a/2)*np.sin(b/2)*np.cos(c/2)

    return np.array([[eta_0, eta_1, eta_2, eta_3]]) 

#definir função que devolve a matriz M
def M(w):
    
    p =  np.radians(w[0])
    q =  np.radians(w[1])   #rad/s
    r =  np.radians(w[2])
    
    return 0.5*np.array([[0, -p, -q, -r],
                     [p, 0, r, -q],
                     [q, -r, 0, p],
                     [r, q, -p, 0]])

#definir função que transforma os valores de eta para os ângulos de euler correspondentes
def X_e(eta):

    phi = np.arctan2(2*(eta[0]*eta[1]+eta[2]*eta[3]), 1-2*((eta[1]**2)+(eta[2]**2)))
    theta = np.arcsin(2*(eta[0]*eta[2]-eta[3]*eta[1]))
    psi = np.arctan2(2*(eta[0]*eta[3]+eta[1]*eta[2]), 1-2*((eta[2]**2)+(eta[3]**2)) )

    return np.array([phi, theta, psi])



#Dados do enunciado:
theta_0 = 89.9
phi_0 = 30
psi_0 = 45
p = 50
q = 30
r = 10
h = 0.01

#definir vetores x_0 e w
x_0 = np.array([phi_0, theta_0, psi_0]) #graus
w = np.array([p, q, r]) #graus

#calcular M e M_d
eta_0 =np.transpose(eta_v0(x_0))    #rad
m = M(w)                            #rad/s
m_d = sp.linalg.expm(m*h)           # F         e^(rad/s) * rad ...

#calcular eta_t1:
eta_1 = m_d.dot(eta_0)              #não sei deve tar bem

#transformar os valores de eta para os ângulos de euler correspondentes:
x_t1 = np.degrees(X_e(eta_1))

print("phi=",x_t1[0])
print("theta=",x_t1[1])
print("psi=",x_t1[2])