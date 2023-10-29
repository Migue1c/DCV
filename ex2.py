import numpy as np
import math

#Exercício 1


#definir valores iniciais de x e y:
x_0 = -2 
y_0 = -3  
#definir passo
x_p = 0.1 
y_p = 0.2
#definir "range" (i_r linhas e j_r colunas)
i_r = (abs(x_0)*2)/x_p    #assumindo que o intervalo é sempre [-a,a]
j_r = (abs(y_0)*2)/y_p    #   "             "
i_r = int(i_r)  #transforma os valores em numeros inteiro, para não dar erro no loop
j_r = int(j_r)

#definir matriz
m = np.eye(i_r,j_r) #cria uma matriz identidade com i_r linhas e j_r colunas

#condições iniciais para os loops
i = 0
j = 0
x = x_0
y = y_0
'''print(i_r)
print(j_r)'''

for i in range(i_r):

    for j in range(j_r):
        m[i,j] = 0.1+((1+np.sin((2*x)+(3*y)))/(3.5+np.sin(x-y)))
        y += y_p
    j = 0
    y = y_0
    x += x_p

'''with open("Matriz.txt", "a") as file:
    np.savetxt(file, m, delimiter="\t", fmt="%f") ''' #para acrescentar os valores da matriz ao ficheiro
np.savetxt("Matriz.txt", m, delimiter="\t", fmt="%f") #dá overwrite ao ficheiro
print(m)