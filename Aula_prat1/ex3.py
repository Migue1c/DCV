import numpy as np
import math

#Exercício 2

#Dados do enunciado:

theta_0 = 89.9
phi_0 = 30
psi_0 = 45
p = 50
q = 30
r = 103
h = 0.01

#definir vetores x_0 e w
x_0 = np.array([phi_0, theta_0, psi_0])
w = np.array([p, q, r])

def eta_v0(x)
    
    eta_0 = np.cos(x[0]/2)*np.cos(x[1]/2)*np.cos(x[2]/2)+np.sin(x[0]/2)*np.sin(x[1]/2)*np.sin(x[2]/2)
    eta_1 = np.sin(x[0]/2)*np.cos(x[1]/2)*np.cos(x[2]/2)-np.cos(x[0]/2)*np.sin(x[1]/2)*np.sin(x[2]/2)
    eta_2 = np.cos(x[0]/2)*np.sin(x[1]/2)*np.cos(x[2]/2)-np.sin(x[0]/2)*np.cos(x[1]/2)*np.sin(x[2]/2)
    eta_3 = np.cos(x[0]/2)*np.cos(x[1]/2)*np.sin(x[2]/2)-np.sin(x[0]/2)*np.sin(x[1]/2)*np.cos(x[2]/2)

