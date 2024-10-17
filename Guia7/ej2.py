import numpy as np
import random 
import matplotlib.pyplot as plt
import math

it = 0 
cant_iteraciones = 300
cant_hormigas = 50 ## k
alpha = 2
beta = 2

D = np.loadtxt('Guia7/gr17.csv', delimiter=',')
n = D.shape

feromonas = np.empty(n,float)


Nk = np.empty((cant_hormigas,n[0]),int)
pk = np.empty(cant_hormigas,object)

## inicializamos las feromonas en un rango chico
for i in range(n[0]):
    for j in range(n[1]):
        feromonas[i,j] = random.uniform(0,0.1)

## inicializamos las hormigas en el nodo cero como origen
for k in range(cant_hormigas):
    Nk[k,0] = 0
    Nk[k,1:] = 1 


#3.repetir
while it < cant_iteraciones :
    #3.1 para cada hormiga
    for k in range(cant_hormigas):
        Nk[k,0] = 0
        Nk[k,1:] = 1 

    for k in range(cant_hormigas):

        #3.1.1
        pk[k] = [0]
        #3.1.2 repetir
        c = 0
        while len(pk[k]) < n[0] : 
           # for i in pk[k]:
            prob = []
            nodo_actual = pk[k][-1]
            ## calculo sumatoria feromonas iu
            suma = 0
            for u in range(n[1]):
                if(Nk[k,u] == 1):
                    suma += (feromonas[nodo_actual,u]**alpha) * ((1/D[nodo_actual,u])**beta)
            
            ## calculo probabilidades
            for j in range(n[0]):
                if(Nk[k,j] == 1):
                    prob.append(float(( (feromonas[nodo_actual,j]**alpha) * (1/D[nodo_actual,j])**beta)/suma))    
                else: 
                    prob.append(0)
                

            ## print(np.sum(prob),c)                        
            indices = np.arange(len(prob))

            nodo_deseado = np.random.choice(indices, p=prob)
            c += 1

            pk[k].append(nodo_deseado)
            Nk[k,nodo_deseado]=0

    print(it)
    it+= 1

