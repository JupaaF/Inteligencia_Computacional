import numpy as np
import random 
import matplotlib.pyplot as plt
import math
import networkx as nx

plt.ion()
# Función para visualizar la mejor solución
def actualizar_grafo(D, mejor_camino, iteracion):
    plt.clf()  # Limpiar el gráfico anterior
    G = nx.Graph()

    # Añadir las ciudades como nodos
    n_cities = D.shape[0]
    for i in range(n_cities):
        G.add_node(i, label=f'Ciudad {i+1}')

    # Añadir las aristas según el mejor camino
    for i in range(len(mejor_camino) - 1):
        G.add_edge(mejor_camino[i], mejor_camino[i + 1], weight=D[mejor_camino[i], mejor_camino[i + 1]])

    pos = nx.spring_layout(G, seed=42)  # Layout para distribuir los nodos

    # Dibujar nodos
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')

    # Dibujar aristas con pesos
    nx.draw_networkx_edges(G, pos, edgelist=G.edges, width=2, edge_color='blue')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    # Etiquetas de nodos
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels=node_labels)

    # Título con número de iteración
    plt.title(f'Mejor camino en la iteración {iteracion}')
    plt.axis('off')
    plt.pause(1)  # Pausar para actualizar el gráfico


### ------------------------------------------------------------------------------ 



it = 0 
cant_iteraciones = 200
cant_hormigas = 30 ## k
alpha = 0.5
beta = 2
Q = 0.2

## D distancias entre ciudades
D = np.loadtxt('Guia7/gr17.csv', delimiter=',')
n = D.shape

feromonas = np.empty(n,float) ##matriz

## Matriz Tabu
## ciudades que podria visitar, 1 o 0
Nk = np.empty((cant_hormigas,n[0]),int)

## Guarda el camino de la hormiga
pk = np.empty(cant_hormigas,object) ## vector de vectores

## constante de dispersión
rho = 0.5

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
    for i in range(n[0]):
        for j in range(n[1]):
            feromonas[i,j] = (1-rho)*feromonas[i,j]
            
            ## delta guarda la sumatoria de los deltas
            for k in range(cant_hormigas):
                for z in range(len(pk[k])-1):
                    ## comprobamos caminos
                    if((pk[k][z] == i) and (pk[k][z+1] == j)):
                        ## calculo delta feromonas
                        ## delta_feromonas[i,j] += Q/dist_hormigas[k]  ## global
                        delta_feromonas[i,j] += Q ## uniforme
                        # delta_feromonas[i,j] += Q ## local                         
                         
            feromonas[i,j] += delta_feromonas[i,j]

#### ------------ graficar -------------------
    if it % 10 == 0:
        mejor_hormiga = np.argmin(dist_hormigas)
        mejor_camino = pk[mejor_hormiga]
        actualizar_grafo(D, mejor_camino, it)
        print(f"mejor distancia: {dist_hormigas[mejor_hormiga]}")

    print(it)
    it+= 1


plt.ioff()
plt.show()