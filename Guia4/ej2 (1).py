import numpy as np
import matplotlib.pyplot as plt
import random
x = np.loadtxt('Guia2\irisbin_trn.csv', delimiter=',')
[cant_filas,cant_columnas]= x.shape
cant_max_it = 1000
k = 3
x = x[:,:-3]
cant_columnas = cant_columnas-3

# print(x)

#### genera matriz de indices random para los k grupos
indices = [[] for _ in range(k)]
for i in range (cant_filas):
    n = random.randrange(k)
    indices[n].append(i)

#### calcula la media de las entradas ubicadas en esos cluster
#### para cada entrada
medias = np.empty((k,cant_columnas),float)
for j in range (k):
    suma = 0
    for i in indices[j]:
        suma += x[i]
    medias[j,:]= suma/len(indices[j])

it=0
while it<cant_max_it:
    it+=1
    #### indices pasa a contener los indices con las distancias mínimas
    indices = [[] for _ in range(k)]
    medias_anterior = medias.copy()
    for i in range (cant_filas):
        dist_min = 10000
        indice_min = 0
        for j in range(k):
            ##calcula distancia entre cada dato y el centroide del cluster definido anteriormente
            ## como la media de los indices aleatorios
            vec_dist = x[i]-medias[j]
            dist = np.dot(vec_dist,vec_dist.T)
            ## se guarda el indice del cluster con la distancia más chica
            ## al centroide
            if(dist<dist_min):
                indice_min = j
                dist_min = dist
        indices[indice_min].append(i)
    for j in range (k):
        suma = 0
        ## calculamos de nuevo el centroide con los indices actualizados
        for i in indices[j]:
            suma += x[i]
        if(len(indices[j])!=0):
            medias[j,:]= suma/len(indices[j])
    aux = medias - medias_anterior

    ## si no hay diferencia entre las medias calculadas y las anteriores sale
    todos_ceros = True
    for fila in aux:
        for elemento in fila:
            if elemento != 0:
                todos_ceros = False
    if(todos_ceros):
        break
        
print(it)

#-----------------GRAFICACION---------------------------

colores = ['r', 'g', 'b', 'c', 'm', 'y']

#------ANCHO Y LONGITUD DEL SEPALO

## defino subplot
fig, axes = plt.subplots(3, 2, figsize=(12, 8))

# Establecer las etiquetas de los ejes x e y
for i in range(3):
    for j in range(2):
        axes[i, j].set_xlabel("Eje X1")
        axes[i, j].set_ylabel("Eje X2")
        # Habilitar la grilla
        axes[i, j].grid(True)

###############################
axes[0, 0].set_title('Medias de clusters respecto al ancho y a la longitud del sepalo')
for j in range(k):
    # Extraer los puntos correspondientes al cluster j
    puntos_cluster = x[indices[j]]
    
    # Graficar los puntos del cluster con un color distinto
    axes[0,0].scatter(puntos_cluster[:, 0], puntos_cluster[:, 1], color=colores[j], label=f'Cluster {j+1}', marker='o')
# Graficar las medias en color azul
axes[0,0].scatter(medias[:, 0], medias[:, 1], color='black', marker='*', s=200, label='Centroides') 


###############################
axes[1, 0].set_title('Medias de clusters respecto a longitud del sepalo y del petalo')
for j in range(k):
    # Extraer los puntos correspondientes al cluster j
    puntos_cluster = x[indices[j]]
    
    # Graficar los puntos del cluster con un color distinto
    axes[1,0].scatter(puntos_cluster[:, 0], puntos_cluster[:, 2], color=colores[j], label=f'Cluster {j+1}', marker='o')
# Graficar las medias en color azul
axes[1,0].scatter(medias[:, 0], medias[:, 2], color='black', marker='*', s=200, label='Centroides') 


###############################
axes[2, 0].set_title('Medias de clusters respecto a la longitud del sepalo y el ancho del petalo')
for j in range(k):
    # Extraer los puntos correspondientes al cluster j
    puntos_cluster = x[indices[j]]
    
    # Graficar los puntos del cluster con un color distinto
    axes[2,0].scatter(puntos_cluster[:, 0], puntos_cluster[:, 3], color=colores[j], label=f'Cluster {j+1}', marker='o')
# Graficar las medias en color azul
axes[2,0].scatter(medias[:, 0], medias[:, 3], color='black', marker='*', s=200, label='Centroides') 


###############################
axes[0, 1].set_title('Medias de clusters respecto al ancho del sepalo y la longitud del petalo')
for j in range(k):
    # Extraer los puntos correspondientes al cluster j
    puntos_cluster = x[indices[j]]
    
    # Graficar los puntos del cluster con un color distinto
    axes[0,1].scatter(puntos_cluster[:, 1], puntos_cluster[:, 2], color=colores[j], label=f'Cluster {j+1}', marker='o')
# Graficar las medias en color azul
axes[0,1].scatter(medias[:, 1], medias[:, 2], color='black', marker='*', s=200, label='Centroides') 

###############################
axes[1,1].set_title('Medias de clusters respecto al ancho del sepalo y el ancho del petalo')
for j in range(k):
    # Extraer los puntos correspondientes al cluster j
    puntos_cluster = x[indices[j]]
    
    # Graficar los puntos del cluster con un color distinto
    axes[1, 1].scatter(puntos_cluster[:, 1], puntos_cluster[:, 3], color=colores[j], label=f'Cluster {j+1}', marker='o')
# Graficar las medias en color azul
axes[1, 1].scatter(medias[:, 1], medias[:, 3], color='black', marker='*', s=200, label='Centroides') 


###############################
axes[2,1].set_title('Gráfico de clusters con sus medias')

for j in range(k):
    # Extraer los puntos correspondientes al cluster j
    puntos_cluster = x[indices[j]]
    
    # Graficar los puntos del cluster con un color distinto
    axes[2,1].scatter(puntos_cluster[:, 2], puntos_cluster[:, 3], color=colores[j], label=f'Cluster {j+1}', marker='o')
# Graficar las medias en color azul
axes[2,1].scatter(medias[:, 2], medias[:, 3], color='black', marker='*', s=200, label='Centroides') 

plt.legend()
plt.tight_layout()
plt.show()