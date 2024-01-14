import numpy as np
from  numpy.linalg import eigvals
import sympy as sp
import matplotlib.pyplot as plt
import scipy as sc
import math as m 
import control as ct

################################################################################################################################################################################
################################################################################################################################################################################

# Exercicio 2.1 

################################################################################################################################################################################
################################################################################################################################################################################

Cm_0 = 0.288
Cm_alfa = -0.135
Cn_beta = 0.0032
Cl_beta = -0.0473
print('Teste: \n')
if Cm_0 > 0 and Cm_alfa < 0:
    print('O aviao e estaticamente estavel longitudinalmente')
else:
    print('O aviao nao e estaticamente estavel longitudinalmente')

if Cn_beta > 0 and Cl_beta < 0:
    print('O aviao e estaticamente estavel latero-direcionalmente')
else:
    print('O aviao nao e estaticamente estavel latero-direcionalmente')

################################################################################################################################################################################
################################################################################################################################################################################

# Exercicio 2.2 

################################################################################################################################################################################
################################################################################################################################################################################

# Dados
h0 = 2000
u0 = 50
u = u0
w0 = 0.95
w = w0
q0 = 0
q = q0

########################################################################################
########################################################################################

#Voo Longitudinal

########################################################################################
########################################################################################
teta0 = 0.153
teta = teta0
CL_0 = 0.050
CL_alpha = 0.079
CL_alpha_ponto = 0.024
CL_q = 0.122
CL_delta_e = 0.322
CD_0 = 0.001
CD_alpha = 0.00010
CD_alpha_ponto = 0
CD_q = 0
CD_delta_e = 0
Cm_0 = 0.288
Cm_alpha = -0.135
Cm_alpha_ponto = -0.112
Cm_q = -0.500
Cm_delta_e = -3.692
epsilon_T =0
delta_e_0 = 0.1
delta_e = delta_e_0 
delta_T_0 = 0.4
delta_T = delta_T_0

#Geometricos/Outros
alpha0 = np.arctan(w0/u0)
alpha = alpha0 
V = np.sqrt(u0**2+ w0**2)
c=0.26
b=1.42
m=2.1
g=9.81
L=m*g
T=10.10
rho_0 =1.225
rho=rho_0*(1-2.2558*10**(-5)*h0)**(4.256060537) 
S=0.43
Ix =0.21
Iy =0.34
Iz =0.53
Ixz =0.02
#alphaponto_0=0
#Q=0.5*rho*V_d**2
h=10**(-5) #Passo para derivadas

# Coeficientes de Estabilidade
#u, alpha, teta, q, delta_T, delta_e, alphaponto = sp.symbols('u alpha teta q delta_T delta_e alphaponto')

def alphaponto(u, alpha, q, teta, delta_T, delta_e):
    alphaponto = q+((g*sp.cos(alpha))/u)*sp.cos(teta-alpha)-(sp.cos(alpha)/(m*u))*(L-delta_T*T*sp.sin(epsilon_T)*sp.cos(alpha)+delta_T*T*sp.cos(epsilon_T)*sp.sin(alpha))
    return alphaponto

#alphap =  alphaponto(u, alpha, q, teta, delta_T, delta_e)
#print(alphap)

def CL(u, alpha, q, teta, delta_T, delta_e):
    C = CL_0+CL_alpha*alpha +(c/(2*V))*( CL_alpha_ponto * alphaponto(u, alpha, q, teta, delta_T, delta_e) +CL_q*q)+ CL_delta_e*delta_e 
    return C

def CD(u, alpha, q, teta, delta_T, delta_e):
    C = CD_0+CD_alpha *alpha +(c/(2*V))*( CD_alpha_ponto * alphaponto(u, alpha, q, teta, delta_T, delta_e) +CD_q*q)+ CD_delta_e*delta_e
    return C

def Cm(u, alpha, q, teta, delta_T, delta_e):
    C = Cm_0+Cm_alpha *alpha +(c/(2*V))*( Cm_alpha_ponto * alphaponto(u, alpha, q, teta, delta_T, delta_e) +Cm_q*q)+ Cm_delta_e*delta_e
    return C

'''
def uponto(u,alpha ,q,teta ,delta_T ,delta_e):
    u = (1/m)*(0.5*rho*S*((u**2)/(np.cos(alpha)**2))*((CL_0+CL_alpha*alpha+(c/(2*V))*
                    (CL_alpha_ponto*alphaponto(u, alpha, q, teta, delta_T, delta_e) + CL_q * q) 
                    +CL_delta_e * delta_e) * np.sin(alpha)-(CD_0 + CD_alpha*alpha+(c/(2*V))*
                    (CD_alpha_ponto*alphaponto(u, alpha, q, teta, delta_T, delta_e) + CD_q * q)+CD_delta_e * delta_e) * np.cos(alpha))
                    + delta_T * T * np.cos(epsilon_T)) - g * np.sin(teta) - q * u * np.tan(alpha)
    return u

'''
def uponto(u, alpha, q, teta, delta_T, delta_e): 
    upoint = (1 / m) * (0.5 * rho * S * (u ** 2) / ((sp.cos(alpha)) ** 2) *
                      ((CL_0 + CL_alpha * alpha + (c / (2 * V)) * (CL_alpha_ponto * alphaponto(u, alpha, q, teta, delta_T, delta_e) + CL_q * q) +
                        CL_delta_e * delta_e) * sp.sin(alpha) -
                       (CD_0 + CD_alpha * alpha + (c / (2 * V)) * (CD_alpha_ponto * alphaponto(u, alpha, q, teta, delta_T, delta_e) + CD_q * q) +
                        CD_delta_e * delta_e) * sp.cos(alpha)) + delta_T * T * sp.cos(epsilon_T)) - g * sp.sin(teta) - q * u * sp.tan(alpha)
    return upoint



def tetaponto(u,alpha ,q,teta ,delta_T ,delta_e):
    return q

def q_ponto(u,alpha ,q,teta ,delta_T ,delta_e):
    qponto = ( (rho *(u**2)*S*c*(Cm(u,alpha ,q,teta ,delta_T ,delta_e)))/(2* Iy*((sp.cos(alpha))**2)))
    return qponto



# Matriz A

#uponto

a11 = (uponto(u0 + h, alpha0, teta0, q0, delta_T_0, delta_e_0) - uponto(u0 - h, alpha0, teta0, q0, delta_T_0, delta_e_0)) / (2 * h)
a12 = (uponto(u0, alpha0 + h, teta0, q0, delta_T_0, delta_e_0) - uponto(u0, alpha0 - h, teta0, q0, delta_T_0, delta_e_0)) / (2 * h)
a13 = (uponto(u0, alpha0, teta0 + h, q0, delta_T_0, delta_e_0) - uponto(u0, alpha0, teta0 - h, q0, delta_T_0, delta_e_0)) / (2 * h)
a14 = (uponto(u0, alpha0, teta0, q0 + h, delta_T_0, delta_e_0) - uponto(u0, alpha0, teta0, q0 - h, delta_T_0, delta_e_0)) / (2 * h)

'''
a11 = (uponto(u + h, alpha, teta, q, delta_T, delta_e) - uponto(u - h, alpha, teta, q, delta_T, delta_e)) / (2 * h)
a12 = (uponto(u, alpha + h, teta, q, delta_T, delta_e) - uponto(u, alpha - h, teta, q, delta_T, delta_e)) / (2 * h)
a13 = (uponto(u, alpha, teta + h, q, delta_T, delta_e) - uponto(u, alpha, teta - h, q, delta_T, delta_e)) / (2 * h)
a14 = (uponto(u, alpha, teta, q + h, delta_T, delta_e) - uponto(u, alpha, teta, q - h, delta_T, delta_e)) / (2 * h)
'''


#alphaponto

a21 =( alphaponto(u0+h,alpha0 ,teta0 ,q0 ,delta_T_0 , delta_e_0 )-alphaponto(u0 -h, alpha0 ,teta0 ,q0 ,delta_T_0 , delta_e_0 ))/(2*h)
a22 =( alphaponto(u0 ,alpha0+h,teta0 ,q0 ,delta_T_0 , delta_e_0 )-alphaponto(u0 ,alpha0 -h,teta0 ,q0 ,delta_T_0 , delta_e_0 ))/(2*h)
a23 =( alphaponto(u0 ,alpha0 ,teta0+h,q0 ,delta_T_0 , delta_e_0 )-alphaponto(u0 ,alpha0 ,teta0 -h,q0 ,delta_T_0 , delta_e_0 ))/(2*h) 
a24 =( alphaponto(u0 ,alpha0 ,teta0 ,q0+h,delta_T_0 , delta_e_0 )-alphaponto(u0 ,alpha0 ,teta0 ,q0 -h,delta_T_0 , delta_e_0 ))/(2*h) 

#q_ponto

a31 =( q_ponto(u0+h,alpha0 ,teta0 ,q0 ,delta_T_0 , delta_e_0 )-q_ponto(u0 -h,alpha0 ,teta0 ,q0 ,delta_T_0 , delta_e_0 ))/(2*h)
a32 =( q_ponto(u0 ,alpha0+h,teta0 ,q0 ,delta_T_0 , delta_e_0 )-q_ponto(u0 ,alpha0 -h,teta0 ,q0 ,delta_T_0 , delta_e_0 ))/(2*h)
a33 =( q_ponto(u0 ,alpha0 ,teta0+h,q0 ,delta_T_0 , delta_e_0 )-q_ponto(u0 ,alpha0 ,teta0 -h ,q0 ,delta_T_0 , delta_e_0 ))/(2*h)
a34 =( q_ponto(u0 ,alpha0 ,teta0 ,q0+h,delta_T_0 , delta_e_0 )-q_ponto(u0 ,alpha0 ,teta0 , q0 -h,delta_T_0 , delta_e_0 ))/(2*h)

#tetaponto

a41 =( tetaponto(u0+h,alpha0 ,teta0 ,q0 ,delta_T_0 , delta_e_0 )-tetaponto(u0 -h,alpha0 ,teta0 ,q0 ,delta_T_0 , delta_e_0 ))/(2*h)
a42 =( tetaponto(u0 ,alpha0+h,teta0 ,q0 ,delta_T_0 , delta_e_0 )-tetaponto(u0 ,alpha0 -h ,teta0 ,q0 ,delta_T_0 , delta_e_0 ))/(2*h)
a43 =( tetaponto(u0 ,alpha0 ,teta0+h,q0 ,delta_T_0 , delta_e_0 )-tetaponto(u0 ,alpha0 , teta0 -h,q0 ,delta_T_0 , delta_e_0 ))/(2*h)
a44 =( tetaponto(u0 ,alpha0 ,teta0 ,q0+h,delta_T_0 , delta_e_0 )-tetaponto(u0 ,alpha0 , teta0 ,q0 -h,delta_T_0 , delta_e_0 ))/(2*h)

AA_long = np.array([[a11, a12, a13, a14], [a21, a22, a23, a24], [a31, a32, a33, a34], [a41, a42, a43, a44]])

# Matriz B

#uponto

b11 =( uponto(u0 ,alpha0 ,teta0 ,q0 , delta_T_0 +h, delta_e_0)-uponto(u0 ,alpha0 ,teta0 , q0 ,delta_T_0 -h, delta_e_0 ))/(2*h)
b12 =( uponto(u0 ,alpha0 ,teta0 ,q0 ,delta_T_0 , delta_e_0 +h)-uponto(u0 ,alpha0 ,teta0 , q0 ,delta_T_0 ,delta_e_0 -h))/(2*h)

#alphaponto

b21 =( alphaponto(u0 ,alpha0 ,teta0 ,q0 , delta_T_0 +h, delta_e_0)-alphaponto(u0 ,alpha0 ,teta0 ,q0 ,delta_T_0 -h, delta_e_0))/(2*h) 
b22 =( alphaponto(u0 ,alpha0 ,teta0 ,q0 ,delta_T_0 , delta_e_0 +h)-alphaponto(u0 ,alpha0 ,teta0 ,q0 ,delta_T_0 ,delta_e_0 -h))/(2*h)

#q_ponto

b31 =( q_ponto(u0 ,alpha0 ,teta0 ,q0 , delta_T_0 +h, delta_e_0)-q_ponto(u0 ,alpha0 ,teta0 , q0 ,delta_T_0-h, delta_e_0 ))/(2*h) 
b32 =( q_ponto(u0 ,alpha0 ,teta0 ,q0 ,delta_T_0 , delta_e_0 +h)-q_ponto(u0 ,alpha0 ,teta0 , q0 ,delta_T_0 ,delta_e_0-h))/(2*h)

#tetaponto

b41 =( tetaponto(u0 ,alpha0 ,teta0 ,q0 , delta_T_0 +h, delta_e_0)-tetaponto(u0 ,alpha0 , teta0 ,q0 ,delta_T_0 -h, delta_e_0 ))/(2*h)
b42 =( tetaponto(u0 ,alpha0 ,teta0 ,q0 ,delta_T_0 , delta_e_0 +h)-tetaponto(u0 ,alpha0 , teta0 ,q0 ,delta_T_0 ,delta_e_0 -h))/(2*h)

BB_long = np.array([[b11, b12], [b21, b22], [b31, b32], [b41, b42]])

A_long = np.array(AA_long, dtype=float)
B_long = np.array(BB_long, dtype=float)


########################################################################################
########################################################################################

#Voo Latero -direcional

########################################################################################
########################################################################################
v0 = 1.1 
v = v0
p0 = 0
p = p0
r0 =0 
r = r0
V_d=np.sqrt(u0**2+ v0**2+ w0**2)
V = V_d
beta0=v0/u0
beta = beta0
fi0 =0.51
delta_a_0 =0.4
delta_a = delta_a_0
delta_r_0 =0.3
delta_r = delta_r_0
Cy_beta = -0.076
Cy_delta_a =0
Cy_delta_r = -0.0023
Cl_beta = -0.0473
Cl_p = -0.3774
Cl_r =0.0021
Cl_delta_a =0.0011
Cl_delta_r =0.0002
Cn_beta =0.0032
Cn_p = -0.0059
Cn_r = -0.0500
Cn_delta_a =0.00001
Cn_delta_r =0.0039
alphaponto_0 =0
Q=0.5* rho*V_d**2
 
alpha=alpha0
q=q0
delta_e= delta_e_0 

#Coeficientes de Estabilidade

#beta, delta_a, delta_r, p, r, fi = sp.symbols('beta delta_a delta_r p r fi')


def Cy(beta , fi , p, r, delta_a , delta_r):
    C = Cy_beta*beta+ Cy_delta_a *delta_a+ Cy_delta_r *delta_r
    return C

def Cl(beta , fi , p, r, delta_a , delta_r):
    C = Cl_beta*beta +(b/(2* V_d))*( Cl_p*p+Cl_r*r)+ Cl_delta_a *delta_a+ Cl_delta_r * delta_r
    return C

def Cn(beta , fi , p, r, delta_a , delta_r):
    C = Cn_beta*beta +(b/(2* V_d))*( Cn_p*p+Cn_r*r)+ Cn_delta_a *delta_a+ Cn_delta_r *delta_r
    return C
'''
Cy=Cy_beta*beta+ Cy_delta_a *delta_a+ Cy_delta_r *delta_r

Cl=Cl_beta*beta +(b/(2* V_d))*( Cl_p*p+Cl_r*r)+ Cl_delta_a *delta_a+ Cl_delta_r * delta_r

Cn=Cn_beta*beta +(b/(2* V_d))*( Cn_p*p+Cn_r*r)+ Cn_delta_a *delta_a+ Cn_delta_r *delta_r
'''
#Funcoes

def vponto (beta , fi , p, r, delta_a , delta_r):
    vponto = (((-Q*S)/m)*(CD(u, alpha, q, teta, delta_T, delta_e)*sp.sin(beta)- Cy(beta , fi , p, r, delta_a , delta_r)*sp.cos(beta))+g*sp.cos(teta0)*sp.sin(fi)+p*w0)-r*u0
    return vponto


def fiponto(beta , fi , p, r, delta_a , delta_r):
    fiponto = p+r*sp.cos(fi)*sp.tan(teta0)
    return fiponto

def pponto(beta , fi , p, r, delta_a , delta_r):
    pponto =((Q*S*b)/(Ix*Iz -( Ixz**2)))*(Iz*Cl(beta , fi , p, r, delta_a , delta_r)+Ixz* Cn(beta , fi , p, r, delta_a , delta_r))
    return pponto


def rponto(beta , fi , p, r, delta_a , delta_r): 
    rponto = ((Q*S*b)/(Ix*Iz -( Ixz**2)))*(Ix*Cn(beta , fi , p, r, delta_a , delta_r)+Ixz* Cl(beta , fi , p, r, delta_a , delta_r))
    return rponto

#Matriz A

#beta_ponto
a_11 =(vponto(beta0+h,fi0 ,p0 ,r0 ,delta_a_0,delta_r_0)-vponto(beta0-h,fi0 ,p0 ,r0 ,delta_a_0,delta_r_0))/(2*h)
a_12 =(vponto(beta0 ,fi0+h,p0 ,r0 ,delta_a_0,delta_r_0)-vponto(beta0,fi0 -h,p0 ,r0 ,delta_a_0,delta_r_0))/(2*h)
a_13 =(vponto(beta0 ,fi0 ,p0+h,r0 ,delta_a_0,delta_r_0)-vponto(beta0,fi0 ,p0 -h,r0 ,delta_a_0,delta_r_0))/(2*h)
a_14 =(vponto(beta0 ,fi0 ,p0 ,r0+h,delta_a_0,delta_r_0)-vponto(beta0,fi0 ,p0 ,r0 -h,delta_a_0,delta_r_0))/(2*h)

#p_ponto
a_21 =( pponto(beta0+h,fi0 ,p0 ,r0 ,delta_a_0 , delta_r_0 )-pponto(beta0 -h,fi0 ,p0 ,r0 ,delta_a_0 , delta_r_0 ))/(2*h)
a_22 =( pponto(beta0 ,fi0+h,p0 ,r0 ,delta_a_0 , delta_r_0 )-pponto(beta0 ,fi0 -h,p0 ,r0 ,delta_a_0 , delta_r_0 ))/(2*h)
a_23 =( pponto(beta0 ,fi0 ,p0+h,r0 ,delta_a_0 , delta_r_0 )-pponto(beta0 ,fi0 ,p0 -h,r0 ,delta_a_0 , delta_r_0 ))/(2*h)
a_24 =( pponto(beta0 ,fi0 ,p0 ,r0+h,delta_a_0 , delta_r_0 )-pponto(beta0 ,fi0 ,p0 ,r0 -h,delta_a_0 , delta_r_0 ))/(2*h)

#r_ponto
a_31 =( rponto(beta0+h,fi0 ,p0 ,r0 ,delta_a_0 , delta_r_0 )-rponto(beta0 -h,fi0 ,p0 ,r0 ,delta_a_0 , delta_r_0 ))/(2*h)
a_32 =( rponto(beta0 ,fi0+h,p0 ,r0 ,delta_a_0 , delta_r_0 )-rponto(beta0 ,fi0 -h,p0 ,r0 ,delta_a_0 , delta_r_0 ))/(2*h)
a_33 =( rponto(beta0 ,fi0 ,p0+h,r0 ,delta_a_0 , delta_r_0 )-rponto(beta0 ,fi0 ,p0 -h,r0 ,delta_a_0 , delta_r_0 ))/(2*h)
a_34 =( rponto(beta0 ,fi0 ,p0 ,r0+h,delta_a_0 , delta_r_0 )-rponto(beta0 ,fi0 ,p0 ,r0 -h,delta_a_0 , delta_r_0 ))/(2*h)

#fi_ponto
a_41 =( fiponto(beta0+h,fi0 ,p0 ,r0 ,delta_a_0 , delta_r_0 )-fiponto(beta0 -h,fi0 ,p0 ,r0,delta_a_0 , delta_r_0 ))/(2*h)
a_42 =( fiponto(beta0 ,fi0+h,p0 ,r0 ,delta_a_0 , delta_r_0 )-fiponto(beta0 ,fi0 -h,p0 ,r0,delta_a_0 , delta_r_0 ))/(2*h)
a_43 =( fiponto(beta0 ,fi0 ,p0+h,r0 ,delta_a_0 , delta_r_0 )-fiponto(beta0 ,fi0 ,p0 -h,r0,delta_a_0 , delta_r_0 ))/(2*h)
a_44 =( fiponto(beta0 ,fi0 ,p0 ,r0+h,delta_a_0 , delta_r_0 )-fiponto(beta0 ,fi0 ,p0 ,r0-h,delta_a_0 , delta_r_0 ))/(2*h)

#montagem 
AA_lat = np.array([[a_11, a_12, a_13, a_14], [a_21, a_22, a_23, a_24], [a_31, a_32, a_33, a_34], [a_41, a_42, a_43, a_44]])

#Matriz B
#beta_ponto
b_11 =( uponto(u0 ,alpha0 ,teta0 ,q0 , delta_T_0 +h, delta_e_0)-uponto(u0 ,alpha0 ,teta0 , q0 ,delta_T_0 -h, delta_e_0))/(2*h)
b_12 =( uponto(u0 ,alpha0 ,teta0 ,q0 ,delta_T_0 , delta_e_0 +h)-uponto(u0 ,alpha0 ,teta0 , q0 ,delta_T_0 ,delta_e_0 -h))/(2*h)

b_21 =( alphaponto(u0 ,alpha0 ,teta0 ,q0 , delta_T_0 +h, delta_e_0)-alphaponto(u0 ,alpha0 ,teta0 ,q0 ,delta_T_0 -h, delta_e_0))/(2*h) 
b_22 =( alphaponto(u0 ,alpha0 ,teta0 ,q0 ,delta_T_0 , delta_e_0 +h)-alphaponto(u0 ,alpha0 ,teta0 ,q0 ,delta_T_0 ,delta_e_0 -h))/(2*h)

#b_21 =( alphaponto(u0 ,alpha0 ,teta0 ,q0 , delta_T_0 +h, delta_e_0)-alphaponto(u0 ,alpha0 ,teta0 ,q0 ,delta_T_0 -h, delta_e_0))/(2*h) 
#b_22 =( alphaponto(u0 ,alpha0 ,teta0 ,q0 ,delta_T_0 , delta_e_0 +h)-alphaponto(u0 ,alpha0 ,teta0 ,q0 ,delta_T_0 ,delta_e_0 -h))/(2*h)

b_31 =( tetaponto(u0 ,alpha0 ,teta0 ,q0 , delta_T_0 +h, delta_e_0)-tetaponto(u0 ,alpha0 , teta0 ,q0 ,delta_T_0 -h, delta_e_0))/(2*h)
b_32 =( tetaponto(u0 ,alpha0 ,teta0 ,q0 ,delta_T_0 , delta_e_0 +h)-tetaponto(u0 ,alpha0 , teta0 ,q0 ,delta_T_0 ,delta_e_0 -h))/(2*h)

b_41 =( q_ponto(u0 ,alpha0 ,teta0 ,q0 , delta_T_0 +h, delta_e_0)-q_ponto(u0 ,alpha0 ,teta0 , q0 ,delta_T_0 -h, delta_e_0 ))/(2*h) 
b_42 =( q_ponto(u0 ,alpha0 ,teta0 ,q0 ,delta_T_0 , delta_e_0 +h)-q_ponto(u0 ,alpha0 ,teta0 , q0 ,delta_T_0 ,delta_e_0 -h))/(2*h)


#Montagem 
BB_lat = np.array([[b_11, b_12], [b_21, b_22], [b_31, b_32], [b_41, b_42]])

A_lat = np.array(AA_lat, dtype=float)
B_lat = np.array(BB_lat, dtype=float)

print("Longitudinal:\n", "A=\n", A_long, "\n","B=\n", B_long, "\n")
print("Latero-direcional:\n", "A=\n", A_lat, "\n","B=\n", B_lat)


########################################################################################
########################################################################################

#tarefa 2.3

########################################################################################
########################################################################################

valores_prop_long= np.linalg.eigvals(A_long)
valores_prop_lat= np.linalg.eigvals(A_lat)



print("\n ################ \n LONGITUDINAL: \n################ \n")
print("Valores proprios long: \n",valores_prop_long)
Con_Long=ct.ctrb(A_long,B_long)
print("Controlabilidade: \n",Con_Long)
cat_Con_Long=np.linalg.matrix_rank(Con_Long)
print("Caracteristica: \n",cat_Con_Long)
C = np.array([[0, 0, 0, 1], 
              [1, 0, 0, 0]])
Ob_long = ct.obsv(A_long,C)
cat_Ob_long = np.linalg.matrix_rank(Ob_long)
print("Matriz de Observabilidade do modelo longitudinal:\n", Ob_long)
print("Rank da Matriz de Observabilidade:", cat_Ob_long)

if cat_Ob_long == 4:
    print("O sistema é completamente observável.")
else:
    print("O sistema é parcialmente observável.")

    

print("\n ################ \n LATERODIRECIONAL: \n################ \n")
print("Valores proprios lat: \n",valores_prop_lat)
Con_Lat=ct.ctrb(A_lat,B_lat)
print("Controlabilidade: \n", Con_Lat)
cat_Con_Lat=np.linalg.matrix_rank(Con_Lat)
print("Caracteristica: \n",cat_Con_Lat)
C = np.array([[0, 0, 0, 1], 
              [1, 0, 0, 0]])
Ob_lat = ct.obsv(A_lat,C)
cat_Ob_lat = np.linalg.matrix_rank(Ob_lat)
print("Matriz de Observabilidade do modelo longitudinal:\n", Ob_lat)
print("Rank da Matriz de Observabilidade:", cat_Ob_lat)

if cat_Ob_lat == 4:
    print("O sistema é completamente observável.")
else:
    print("O sistema é parcialmente observável.")

################################################################################################################################################################################
################################################################################################################################################################################

#tarefa 3

################################################################################################################################################################################
################################################################################################################################################################################

print("\n ################ \n LONGITUDINAL: \n################ \n")

#LONGITUDINAL
u_max =65
w_max =1.7
q_max =3.14
theta_max = 0.5
delta_e_max =0.45
delta_T_max =1
alpha_max = np.arctan(w_max/u_max)
#LATERO-DIRECIONAL
v_max = 1.5
p_max = 3.14
r_max = 3.14
phi_max = 0.8
delta_a_max = 0.35
delta_r_max = 0.52



#5<lambda<100    0.01<miu<0.98
lambd = 90
miu = 0.9
########################################################################################
########################################################################################
    
#LONGITUDINAL

########################################################################################
########################################################################################
#Coeficientes:
lambda_Long=80
miu_Long=1

#Valores de referência:
theta_ref= 0.151
u_ref= 42

#Funções para obter K:
R_Long=np.array([[miu_Long/(delta_e_max**2), 0],[0, miu_Long/(delta_T_max**2)]])
Q_Long=np.array([[lambda_Long/(u_max**2), 0, 0, 0],[0, lambda_Long/(w_max**2), 0, 0],[0, 0, lambda_Long/(q_max**2), 0],[ 0, 0, 0, lambda_Long/(theta_max**2)]])

[K_Long,P_Long,E_Long]=ct.lqr(A_long,B_long,Q_Long,R_Long)
print("K=\n",K_Long)
print("P=\n",P_Long)
print("E=\n",E_Long)


#Controlador para a Origem:
k=0
t0=0
tf=20
delta_t=0.01

n=int(tf/delta_t)
theta_k0=np.zeros((n+2,1))
u_k0=np.zeros((n+2,1))
vt=np.zeros((n+2,1))
tk0=t0
Ac=A_long-B_long@K_Long
Ad=sc.linalg.expm(Ac*delta_t)
x_0=np.array([[42],[alpha0],[0],[0.151]])
xk0=x_0
theta_k0[k]=xk0[3]
u_k0[k]=xk0[0]

while tk0<=tf:
    xk0=Ad@xk0
    #uk=K_Long@xk
    k=k+1
    tk0=tk0+delta_t
    vt[k]=tk0
    theta_k0[k]=xk0[3]
    u_k0[k]=xk0[0]
  

#Controlador para fora da Origem:
    #determinar valores de referência
M_eq = np.hstack((np.vstack((A_long,C)),np.vstack((B_long,np.zeros((2,2))))))
M_eq_inv = np.linalg.inv(M_eq)
y_ref = np.array([0,0,0,0,theta_ref,u_ref])
ref = M_eq_inv@y_ref.T
x_ast=np.array([[ref[0]],[ref[1]],[ref[2]],[ref[3]]])
u_ast=np.array([[ref[4]],[ref[5]]])

k=0
theta_k=np.zeros((n+2,1))
u_k=np.zeros((n+2,1))
vt=np.zeros((n+2,1))

tk=t0
#simulação:
Ac=A_long-B_long@K_Long
Ad=sc.linalg.expm(Ac*delta_t)
M = B_long@(u_ast+K_Long @ x_ast)
x_0=np.array([[u_max],[alpha_max],[q_max],[theta_max]])
xk=x_0
theta_k[k]=xk[3]
u_k[k]=xk[0]


while tk<=tf:
    xk=Ad@(xk+(np.linalg.inv(Ac)@M))-np.linalg.inv(Ac)@M
    k=k+1
    tk=tk+delta_t
    vt[k]=tk
    theta_k[k]=xk[3]
    u_k[k]=xk[0]



fig, axs = plt.subplots(2, 1, figsize=(8, 6))

# Plotar o primeiro gráfico no primeiro subplot
axs[0].plot(vt, theta_k0,label='theta')

axs[0].legend()

# Plotar o segundo gráfico no segundo subplot
axs[1].plot(vt,u_k0 ,label='u')
axs[1].legend()

# Ajustar o layout para evitar sobreposição
plt.tight_layout()
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

# Plotar o primeiro gráfico no primeiro subplot
axs[0].plot(vt, theta_k,label='theta')

axs[0].legend()

# Plotar o segundo gráfico no segundo subplot
axs[1].plot(vt,u_k ,label='u')
axs[1].legend()

# Ajustar o layout para evitar sobreposição
plt.tight_layout()

# Exibir a figura
plt.show()


########################################################################################
########################################################################################
    
#LATERO-DIRECIONAL

########################################################################################
########################################################################################

print("\n ################ \n LATERODIRECIONAL: \n################ \n")

#Coeficientes:
lambda_lat=80
miu_lat=1

#Valores de referência:
theta_ref= 0.151
u_ref= 42

#Funções para obter K:
R_lat=np.array([[miu_lat/(delta_e_max**2), 0],[0, miu_lat/(delta_T_max**2)]])
Q_lat=np.array([[lambda_lat/(u_max**2), 0, 0, 0],[0, lambda_lat/(w_max**2), 0, 0],[0, 0, lambda_lat/(q_max**2), 0],[ 0, 0, 0, lambda_lat/(theta_max**2)]])

[K_lat,P_lat,E_lat]=ct.lqr(A_lat,B_lat,Q_lat,R_lat)
print("K=\n",K_lat)
print("P=\n",P_lat)
print("E=\n",E_lat)


#Controlador para a Origem:
k=0
t0=0
tf=50
delta_t=0.01

n=int(tf/delta_t)
phi_k0=np.zeros((n+2,1))
v_k0=np.zeros((n+2,1))
vt=np.zeros((n+2,1))
tk=t0
Ac=A_lat-B_lat@K_lat
Ad=sc.linalg.expm(Ac*delta_t)
x_0=np.array([[v0],[0],[0],[0.53]])
xk0=x_0
phi_k0[k]=xk0[3]
v_k0[k]=xk0[0]

while tk<=tf:
    xk0=Ad@xk0
    #uk=K_Long@xk
    k=k+1
    tk=tk+delta_t
    vt[k]=tk
    phi_k0[k]=xk0[3]
    v_k0[k]=xk0[0]
  
#Controlador para fora da Origem:
    #determinar valores de referência
M_eq = np.hstack((np.vstack((A_lat,C)),np.vstack((B_lat,np.zeros((2,2))))))
M_eq_inv = np.linalg.inv(M_eq)
y_ref = np.array([0,0,0,0,fi0,v0])
ref = M_eq_inv@y_ref.T
x_ast=np.array([[ref[0]],[ref[1]],[ref[2]],[ref[3]]])
u_ast=np.array([[ref[4]],[ref[5]]])

k=0
phi_k=np.zeros((n+2,1))
v_k=np.zeros((n+2,1))
vt=np.zeros((n+2,1))

tk=t0
#simulação:
Ac=A_lat-B_lat@K_lat
Ad=sc.linalg.expm(Ac*delta_t)
M = B_lat@(u_ast+K_lat @ x_ast)
x_0=np.array([[v_max],[p_max],[r_max],[phi_max]])
xk=x_0
phi_k[k]=xk[3]
v_k[k]=xk[0]
while tk<=tf:
    xk=Ad@(xk+(np.linalg.inv(Ac)@M))-np.linalg.inv(Ac)@M
    k=k+1
    tk=tk+delta_t
    vt[k]=tk
    phi_k[k]=xk[3]
    v_k[k]=xk[0]


#Graficos para a origem
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

# Plotar o primeiro gráfico no primeiro subplot
axs[0].plot(vt, phi_k0,label='phi')

axs[0].legend()

# Plotar o segundo gráfico no segundo subplot
axs[1].plot(vt,v_k0 ,label='v')
axs[1].legend()

# Ajustar o layout para evitar sobreposição
plt.tight_layout()

#Graficos fora da origem
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

# Plotar o primeiro gráfico no primeiro subplot
axs[0].plot(vt, phi_k,label='phi')

axs[0].legend()

# Plotar o segundo gráfico no segundo subplot
axs[1].plot(vt,v_k ,label='v')
axs[1].legend()

# Ajustar o layout para evitar sobreposição
plt.tight_layout()

# Exibir a figura
plt.show()