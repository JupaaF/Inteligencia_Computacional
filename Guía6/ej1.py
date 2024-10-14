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
indices_ord = [i for i, _ in sorted(enumerate(vector_fen), key=lambda x: x[1])]


#Repetir hasta cumplir aptitud
cant_iteraciones = 20
it = 0
while(it < cant_iteraciones):
 #Generar nueva poblacion
     #seleccionar padres (metodo de ventana)
    elite = poblacion[indices_ord[0]]
    for i in range(cant_individuos): #cantidad de ventanas 
        
        
            #a) ruelta:  f = [4.5 2 3 0.5] divido sobre la suma
            #          =[0.45 0.2 0.3 0.05]
            # => hago sumo hasta cada uno: [0.45 0.6 0.95 1]
            #      tiro un nro random entre 0 y 1 y si es 0<x<0.45, elijo el primero, etc etc
            #b) ventana:  ordeno por fitness, elijo ventana, saco uno aleat de ese rango
            #              y repito para otra ventana pequeña (puede salir 2 o más veces el mismo individio)   
            #c) competencia:
        #cruzas
        #reemplazamos toda la poblacion
        #mutacion a todos los indiv (probabilidad muy baja)
    #evaluar fitness y guardar valores



