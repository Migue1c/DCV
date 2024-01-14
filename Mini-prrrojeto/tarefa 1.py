import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


#function to compute eta values:
def eta_v0(x):
    #a = np.radians(x[0])    #phi
    #b= np.radians(x[1])    #theta
    #c = np.radians(x[2])    #psi
    a = x[0] #phi
    b = x[1] #theta
    c = x[2] #psi

    eta_0 = np.cos(a/2)*np.cos(b/2)*np.cos(c/2)+np.sin(a/2)*np.sin(b/2)*np.sin(c/2)
    eta_1 = np.sin(a/2)*np.cos(b/2)*np.cos(c/2)-np.cos(a/2)*np.sin(b/2)*np.sin(c/2)
    eta_2 = np.cos(a/2)*np.sin(b/2)*np.cos(c/2)+np.sin(a/2)*np.cos(b/2)*np.sin(c/2)
    eta_3 = np.cos(a/2)*np.cos(b/2)*np.sin(c/2)-np.sin(a/2)*np.sin(b/2)*np.cos(c/2)

    eta_v = np.array([  [eta_0] ,
                        [eta_1] , 
                        [eta_2] , 
                        [eta_3] ]) 

    return eta_v

#function to compute M matrix:
def M(pqr):
    
    p =  pqr[0]
    q =  pqr[1]  #rad/s
    r =  pqr[2]
    
    m = 0.5*np.array([[0, -p, -q, -r],
                     [p, 0, r, -q],
                     [q, -r, 0, p],
                     [r, q, -p, 0]])
    
    return m

#function to convert eta_dot values back to euler angles:
def X_e(eta_u):
    eta = np.array([eta_u[0,0], eta_u[1,0], eta_u[2,0], eta_u[3,0]])
    phi = np.arctan2(2*(eta[0]*eta[1]+eta[2]*eta[3]), 1-(2*((eta[1]**2)+(eta[2]**2))))
    theta = np.arcsin(2*(eta[0]*eta[2]-eta[3]*eta[1]))
    psi = np.arctan2(2*(eta[0]*eta[3]+eta[1]*eta[2]), 1-(2*((eta[2]**2)+(eta[3]**2))))

    eta_1 = np.array([phi, theta, psi])

    return eta_1



#File reader
dataf_read     = ['tempo','p','q','r']
dataf_pqr         = pd.read_excel('AnexoA.xls', sheet_name = 'Sheet14', usecols = dataf_read, skiprows = 5)

#print(dataf_pqr) 
    
dataf_read1 = ['fi0', 'teta0', 'psi0'] #trocar a ordem das colunas no excel
dataf_ang = pd.read_excel('AnexoA.xls', sheet_name = 'Sheet14', usecols = dataf_read1, skiprows = 1, nrows=1)
#print(df_ang)

ang_0 = np.array(dataf_ang) 
ang_0 = ang_0.reshape(-1)
print(ang_0)
valores_t = np.array(dataf_pqr) # t ; p ; q ; r
#print(valores_t)

h = valores_t[1,0] - valores_t[0,0]
#print(h)

#matrix to store values
x_r = np.zeros((len(valores_t),4))
#initialization
ang_i = ang_0
i = 0
while i < len(valores_t):
    #store values
    x_r[i,0] = valores_t[i,0]
    x_r[i,1] = ang_i[0]
    x_r[i,2] = ang_i[1]
    x_r[i,3] = ang_i[2]
    #compute current iteration
    eta_i = eta_v0(ang_i)
    pqr = np.array([valores_t[i,1], valores_t[i,2], valores_t[i,3]])
    m_i = M(pqr)
    m_d = sp.linalg.expm(m_i * h)
    eta_1 = m_d @ eta_i
    #print(eta_1)
    #values for next iteration
    ang_i = X_e(eta_1) #value will be stored on the next iteration
    i += 1
#print(x_r)

#convert matrix to dataframe (easier plotting)
x_dataf = pd.DataFrame(x_r, columns = ['tempo', 'phi', 'theta', 'psi'])
print(x_dataf)

x_dataf.plot(x='tempo',y=['phi', 'theta', 'psi'])
plt.show()