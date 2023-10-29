import numpy as np
import math

#Exercício 1


#definir valores iniciais de x e y:
x = -2 
y = -3  
#definir passo
x_p = 0.1 
y_p = 0.2
#definir "range" (i_r linhas e j_r colunas)
i_r = (abs(x)*2)/x_p    #assumindo que o intervalo é sempre [-a,a]
j_r = (abs(y)*2)/y_p    #   "             "
i_r = int(i_r)  #transforma os valores em numeros inteiro, para não dar erro no loop
j_r = int(j_r)
#condições iniciais para os loops
i = 0
j = 0
#definir matriz
m = np.eye(i_r,j_r) #cria uma matriz identidade com i_r linhas e j_r colunas

'''print(i_r)
print(j_r)'''

for i in range(i_r):

    for j in range(j_r):
        m[i,j] = 0.1+((1+np.sin((2*x)+(3*y)))/(3.5+np.sin(x-y)))
        y += y_p
    j = 0
    x += x_p

print(m)