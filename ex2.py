import numpy as np
import math

#Exercício 1:


#definir valores iniciais de x e y:
x_0 = -2 
y_0 = -3  
#definir passo:
x_p = 0.1 
y_p = 0.2
#definir "range" (i_r linhas e j_r colunas):
i_r = (abs(x_0)*2)/x_p    #assumindo que o intervalo é sempre [-a,a]
j_r = (abs(y_0)*2)/y_p    #   "             "
i_r = int(i_r)  #transforma os valores em numeros inteiros, para não dar erro no loop
j_r = int(j_r)

#definir matriz:
m = np.eye(i_r,j_r) #cria uma matriz identidade com i_r linhas e j_r colunas

#condições iniciais para os loops:
i = 0
j = 0
x = x_0
y = y_0
#print(i_r)
#print(j_r)

#loop para atribuir os valores à matriz:
for i in range(i_r):

    for j in range(j_r):
        m[i,j] = 0.1+((1+np.sin((2*x)+(3*y)))/(3.5+np.sin(x-y)))
        y += y_p
    j = 0
    y = y_0
    x += x_p

#escrever a matriz num ficheiro de texto:

'''with open("Matriz.txt", "a") as file:
    np.savetxt(file, m, delimiter="\t", fmt="%f") ''' #para acrescentar os valores da matriz ao ficheiro, não é necessário neste caso

np.savetxt("Matriz.txt", m, delimiter="\t", fmt="%f") #dá overwrite ao ficheiro
print(m)


#Exercício 2:

#reset das condições iniciais:
i = 0
j = 0
#definir uma variável para ser o menor numero e atribuir um valor inicial:
a = m[0,6] 
#definir uma variável para ser as "coordenadas" desse numero:
a_c =np.array([0,0])

#loop que dá "scan" aos elementos da matriz (6 coluna) e deteta qual o menor:
for i in range(i_r):
    if m[i,6] <= a:
        a = m[i,6]
        k = i+1
        a_c = np.array([k,6])

    j=0

print("2.O menor elemento da sexta coluna é:",a_c,"com o valor:",a)



#Exercício 3:

#reset das condições iniciais
i = 0
j = 0
#definir uma variável para ser o menor numero e atribuir um valor inicial:
b = m[0,0] 
#definir uma variável para ser as "coordenadas" desse numero:
b_c =np.array([0,0])

#loop que dá "scan" aos elementos da matriz e deteta qual o menor:
for i in range(i_r):

    for j in range(j_r):
        if m[i,j] <= b:
            b = m[i,j]
            k1 = i+1
            n1 = j+1
            b_c = np.array([k1,n1])

    j=0

print("3.O menor elemento da Matriz é:",b_c,"com o valor:",b)



#Exercício 4:

#reset das condições iniciais:
i = 0
j = 0
#definir uma variável para ser o maior numero e atribuir um valor inicial:
c = m[0,0] 
#definir uma variável para ser as "coordenadas" desse numero:
c_c =np.array([0,0])

#loop que dá "scan" aos elementos da matriz e deteta qual o maior:
for i in range(i_r):

    for j in range(j_r):
        if m[i,j] >= c:
            c = m[i,j]
            k2 = i+1
            n2 = j+1
            c_c = np.array([k2,n2])

    j=0

print("4.O maior elemento da Matriz é:",c_c,"com o valor:",c)
