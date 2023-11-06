import numpy as np
import math

#1.
#definir o vetor original:
Cg_P = np.array([[120],
                [130],
                [140]])

#definir valores para os ângulos:
alpha = math.radians(5)
beta = math.radians(1)
theta = math.radians(60)
phi = math.radians(30)
psi = math.radians(45)

#definir matriz de transformação [T] (como uma função):
def T(a,b): #a->alpha e b->beta
    return np.array([[math.cos(a)*math.cos(b), -math.cos(a)*math.sin(b), -math.sin(a)],  
                     [math.sin(b), math.cos(b), 0],
                     [math.sin(a)*math.cos(b), -math.sin(a)*math.sin(b), math.cos(a)]])

#resultado:
Cg_P_aeronave = T(alpha,beta).dot(Cg_P)
print("Ex1.1 - Cg_P(Rb):",Cg_P_aeronave[0,0],"Xb",Cg_P_aeronave[1,0],"Yb",Cg_P_aeronave[2,0],"Zb")



#2.
#definir o vetor 0_Cg
O_Cg = np.array([[5],
                [6],
                [7.5]])*(10**3)

#tranformar Cg_P(Rb) para Cg_P(R0)

#definir matriz de tranformação [R]
def R(a,b,c): #a->theta  b->phi  c->psi
    return np.array([[math.cos(c)*math.cos(a), math.sin(c)*math.cos(a), -math.sin(a)],
                     [-math.sin(c)*math.cos(b)+math.cos(c)*math.sin(b)*math.sin(a), math.cos(c)*math.cos(b)+math.sin(c)*math.sin(b)*math.sin(a), math.cos(a)*math.sin(b)],
                     [math.sin(c)*math.sin(b)+math.cos(c)*math.cos(b)*math.sin(a), -math.cos(c)*math.sin(b)+math.sin(c)*math.cos(b)*math.sin(a), math.cos(a)*math.cos(b)]])

#transformar o ponto para o referencial R0:
Cg_P_terrestre = R(theta,phi,psi).dot(Cg_P_aeronave)
#print(Cg_P_terrestre)

#o vetor O_P é a soma dos vetores O_Cg e Cg_P (no referencial terrestre)
O_P = O_Cg + Cg_P_terrestre
#print(O_P)
print("Ex1.2 - O_P(Ro):",O_P[0,0],"Xo",O_P[1,0],"Yo",O_P[2,0],"Zo") 1