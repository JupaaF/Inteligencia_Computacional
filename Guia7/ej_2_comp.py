import numpy as np
import random 
import math
import pandas as pd


#ind_feromonas = 0 = global
#ind_feromonas = 1 = uniforme
#ind_feromonas = 2 = local
def sistema_hormigas(cant_iteraciones,cant_hormigas,alpha,beta,Q,rho,ind_feromonas,num_iter_sin_mejora):

    it = 0 
    ## D distancias entre ciudades
    D = np.loadtxt('Guia7/gr17.csv', delimiter=',')
    n = D.shape
    mejor_distancia_global = 4000

    feromonas = np.empty(n,float) ##matriz
    iteraciones_sin_mejora = 0
    

    ## Matriz Tabu
    ## ciudades que podria visitar, 1 o 0
    Nk = np.empty((cant_hormigas,n[0]),int)

    ## Guarda el camino de la hormiga
    pk = np.empty(cant_hormigas,object) ## vector de vectores


    ## inicializamos las feromonas en un rango chico
    for i in range(n[0]):
        for j in range(n[1]):
            feromonas[i,j] = random.uniform(0,0.1)


    #3.repetir
    while it < cant_iteraciones :
        #3.1 para cada hormiga

        ## vector distancias hormigas
        ## reinicializamos las distancias
        dist_hormigas = np.zeros(cant_hormigas)
        
        ## suma de los deltas 
        delta_feromonas = np.zeros(n) 
        ## inicializamos las hormigas en el nodo cero como origen
        for k in range(cant_hormigas):
            Nk[k,0] = 0
            Nk[k,1:] = 1 

        for k in range(cant_hormigas):

            #3.1.1
            pk[k] = [0]
            #3.1.2 repetir

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
                    
                ## selecciona de manera aleatoria dependiendo de la probabilidad calculada
                indices = np.arange(len(prob))
                nodo_deseado = np.random.choice(indices, p=prob)

                pk[k].append(nodo_deseado)
                Nk[k,nodo_deseado]=0

            pk[k].append(0)

            ## 3.1.3
            ## calculo de la distancia del camino de las hormigas
            ## recorremos las distancias matriz D con los indices
            ## de pk
            for i in range(len(pk[k])-1):
                dist_hormigas[k] += D[pk[k][i],pk[k][i+1]] 


        ## 3.2             
        ## delta guarda la sumatoria de los deltas
        for i in range(n[0]):
            for j in range(n[1]):
                feromonas[i,j] = (1-rho)*feromonas[i,j]
                
                ## delta guarda la sumatoria de los deltas
                for k in range(cant_hormigas):
                    for z in range(len(pk[k])-1):
                        ## comprobamos caminos
                        if((pk[k][z] == i) and (pk[k][z+1] == j)):
                            ## calculo delta feromonas
                            if(ind_feromonas == 0):
                                delta_feromonas[i,j] += Q/dist_hormigas[k]  ## global
                            if(ind_feromonas == 1):
                                delta_feromonas[i,j] += Q ## uniforme
                            if(ind_feromonas == 2):
                                delta_feromonas[i,j] += Q/D[i,j] ## local                         
                            
                feromonas[i,j] += delta_feromonas[i,j]

        # print(it)
        # print(f'Mejor dist: {np.min(dist_hormigas)}')

        # Inicializamos las variables para controlar el proceso
        mejor_distancia = np.min(dist_hormigas)  # Mejor distancia inicial

        if mejor_distancia < mejor_distancia_global:
            mejor_distancia_global = mejor_distancia
            iteraciones_sin_mejora = 0  # Resetear contador si hay mejora
        else:
            iteraciones_sin_mejora += 1  # Aumentar el contador si no mejora

        # Criterio de parada por iteraciones consecutivas sin mejora
        if iteraciones_sin_mejora >= num_iter_sin_mejora:
            # print(f"Algoritmo detenido en la iteración {it+1} por falta de mejoras.")
            # print(f'Mejor dist: {mejor_distancia_global} encontrado en it {it-num_iter_sin_mejora}')
            break
            
        it+= 1
    return (it-num_iter_sin_mejora),mejor_distancia_global

# LLamados a la funcion
#sistema_hormigas(cant_iteraciones,cant_hormigas,alpha,beta,Q,rho,ind_feromonas)
cant_iteraciones = 100
num_iter_sin_mejora = 25
cant_hormigas = 20
alpha = 1
beta = 2
Q = 0.5

resultados = []
numero_pruebas = 10
tasas_evaporacion = [0.1, 0.5, 0.9]

for metodo in [0,1,2]:
    for tasa in tasas_evaporacion:
        distancias = []
        iteraciones = []
        for t in range(numero_pruebas):
            iteracion_min,distancia_minima = sistema_hormigas(cant_iteraciones,cant_hormigas,alpha,beta,Q,tasa,metodo,num_iter_sin_mejora)
            distancias.append(distancia_minima)
            iteraciones.append(iteracion_min)
            print(t)

        media_dist = np.mean(distancias)    
        var_dist = np.var(distancias)
        media_it = np.mean(iteraciones)    
        var_it = np.var(iteraciones)

        resultados.append([metodo, tasa, media_dist, var_dist, media_it, var_it])     

df = pd.DataFrame(resultados, columns=["Método de Feromonas", "Tasa de Evaporación", "Media Distancia Mínima","Var Distancia Mínima", "Media Iteración", "Var Iteración"])
print(df)

