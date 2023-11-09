import numpy as np
import math
import scipy.linalg as sl

#Exercício 2

#Dados do enunciado:

theta_0 = 89.9
phi_0 = 30
psi_0 = 45
p = 50
q = 30
r = 10
h = 0.01

#definir vetores x_0 e w
x_0 = np.array([phi_0, theta_0, psi_0])
w = np.array([p, q, r])

#definir função que devolve o vetor correspondente de quaterniões
def eta_v0(x):
    
    eta_0 = np.cos(x[0]/2)*np.cos(x[1]/2)*np.cos(x[2]/2)+np.sin(x[0]/2)*np.sin(x[1]/2)*np.sin(x[2]/2)
    eta_1 = np.sin(x[0]/2)*np.cos(x[1]/2)*np.cos(x[2]/2)-np.cos(x[0]/2)*np.sin(x[1]/2)*np.sin(x[2]/2)
    eta_2 = np.cos(x[0]/2)*np.sin(x[1]/2)*np.cos(x[2]/2)-np.sin(x[0]/2)*np.cos(x[1]/2)*np.sin(x[2]/2)
    eta_3 = np.cos(x[0]/2)*np.cos(x[1]/2)*np.sin(x[2]/2)-np.sin(x[0]/2)*np.sin(x[1]/2)*np.cos(x[2]/2)

    return np.array([eta_0, eta_1, eta_2, eta_3])

#definir função que devolve a matriz M
def M(w):
    
    p = w[0]
    q = w[1]
    r = w[2]
    
    return 0.5*np.array([[0, -p, -q, -r],
                     [p, 0, r, -q],
                     [q, -r, 0, p],
                     [r, q, -p, 0]])


