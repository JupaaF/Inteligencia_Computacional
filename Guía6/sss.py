import numpy as np
import random 

cant_bits = 20

x1 = -512
x2 = 512
cant_individuos = 15

def gen_fen(x):
    sum = 0
    for i in range (len(x)):
        sum += (2**(len(x)-i-1))*x[i]
    
    sum = x1 + sum/(x2-x1)
    
    return sum

def fitness(x):
    y = -x*np.sin(np.sqrt(np.abs(x)))
    return -y


#Inicializar población --> 1) crear individuos como cadenas de bits
poblacion = np.empty((cant_individuos,cant_bits),int)

for i in range(cant_individuos):
    for j in range(cant_bits):
        poblacion[i,j] = random.randint(0,1)

#Evaluar-->1) traducir cadena de bit a parametros    
vector_fen = np.empty(cant_individuos,float)
for i in range(cant_individuos):
    vector_fen[i] = gen_fen(poblacion[i,:])


# -->2) evaluar y guardar valores fitness
f = fitness(vector_fen)

indices_ord = [i for i, _ in sorted(enumerate(vector_fen), key=lambda x: x[1], reverse=True)]

for i in range (len(indices_ord)):
    print(f"{vector_fen[indices_ord[i]]}")
