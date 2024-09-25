import numpy as np
import matplotlib.pyplot as plt
import random
x = np.loadtxt('Guia4/circulo.csv', delimiter=',')
[cant_filas,cant_columnas]= x.shape
cant_max_it = 1000
k = 4

indices = [[] for _ in range(k)]
for i in range (cant_filas):
    n = random.randrange(k)
    indices[n].append(i)

medias = np.empty((k,cant_columnas),float)
for j in range (k):
    suma = 0
    for i in indices[j]:
        suma += x[i]
    medias[j,:]= suma/len(indices[j])

it=0
while it<cant_max_it:
    it+=1
    indices = [[] for _ in range(k)]
    medias_anterior = medias.copy()
    for i in range (cant_filas):
        dist_min = 10000
        indice_min = 0
        for j in range(k):
            vec_dist = x[i]-medias[j]
            dist = np.dot(vec_dist,vec_dist.T)
            if(dist<dist_min):
                indice_min = j
                dist_min = dist
        indices[indice_min].append(i)
    for j in range (k):
        suma = 0
        for i in indices[j]:
            suma += x[i]
        if(len(indices[j])!=0):
            medias[j,:]= suma/len(indices[j])
    aux = medias - medias_anterior

    todos_ceros = True
    for fila in aux:
        for elemento in fila:
            if elemento != 0:
                todos_ceros = False
    if(todos_ceros):
        break
        
print(it)
colores = ['r', 'g', 'b', 'c', 'm', 'y']
plt.title('Gráfico de clusters con sus medias')
plt.xlabel('Eje X1')
plt.ylabel('Eje X2')
plt.grid(True)

for j in range(k):
    # Extraer los puntos correspondientes al cluster j
    puntos_cluster = x[indices[j]]
    
    # Graficar los puntos del cluster con un color distinto
    plt.scatter(puntos_cluster[:, 0], puntos_cluster[:, 1], color=colores[j], label=f'Cluster {j+1}', marker='o')

# Graficar las medias en color azul
plt.scatter(medias[:, 0], medias[:, 1], color='black', marker='*', s=200, label='Centroides') 

plt.legend()
plt.show()