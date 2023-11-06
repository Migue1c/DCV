import numpy as np
import math

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

#definir matriz de transformação (como uma função):
def T(a,b): #a->alpha e b->beta
    return np.array([[math.cos(a)*math.cos(b), -math.cos(a)*math.sin(b), -math.sin(a)],  
                     [math.sin(b), math.cos(b), 0],
                     [math.sin(a)*math.cos(b), -math.sin(a)*math.sin(b), math.cos(a)]])

#resultado:
Cg_P_aeronave = T(alpha,beta).dot(Cg_P)
print("Cg_P(Rb):",Cg_P_aeronave[0],"Xb",Cg_P_aeronave[1],"Yb",Cg_P_aeronave[2],"Zb")
