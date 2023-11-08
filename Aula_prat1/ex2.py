import numpy as np
import math

#Exercício 2

#Dados do enunciado:

theta_0 = 20
phi_0 = 30
psi_0 = 45
p = 50
q = 30
r = 10
h = 0.01

#definir vetores x_0 e w
x_0 = np.array([phi_0, theta_0, psi_0])
w = np.array([p, q, r])

#definir uma função que devolve os valores das derivadas de phi, theta e psi:
def f(x,w):
    #definir variaveis
    p = w[0]
    q = w[1]
    r = w[2]
    phi = np.radians(x[0])
    theta = np.radians(x[1])
    psi = np.radians(x[2])
    #valores das derivadas
    phi_d = p + (q*math.sin(phi)+r*math.cos(phi))*math.tan(theta)
    theta_d = q*np.cos(phi)-r*np.sin(phi)
    psi_d = (q*np.sin(phi)+r*np.cos(phi))/(np.cos(theta))
    #devolver vetor com os valores das derivadas
    return np.array([phi_d, theta_d, psi_d])

#método de euler modificado
k_1 = f(x_0,w)
z_0 = x_0 + k_1*h
#print(z_0)
k_2 = f(z_0,w)

x_1 = x_0 + h*((k_1 + k_2)/2)
print(x_1)