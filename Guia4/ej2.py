import numpy as np
import matplotlib.pyplot as plt
import random
x = np.loadtxt('Guia2\irisbin_trn.csv', delimiter=',')
[cant_filas,cant_columnas]= x.shape
cant_max_it = 1000
k = 3
x = x[:,:-3]
cant_columnas = cant_columnas-3
print(x)
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

#-----------------GRAFICACION---------------------------

colores = ['r', 'g', 'b', 'c', 'm', 'y']

#------ANCHO Y LONGITUD DEL SEPALO

plt.figure(1)
plt.title('Gráfico de clusters con sus medias respecto al ancho y a la longitud del sepalo')
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


plt.figure(2)
plt.title('Gráfico de clusters con sus medias respecto a la longitud del sepalo y la longitud del petalo')
plt.xlabel('Eje X1')
plt.ylabel('Eje X2')
plt.grid(True)
for j in range(k):
    # Extraer los puntos correspondientes al cluster j
    puntos_cluster = x[indices[j]]
    
    # Graficar los puntos del cluster con un color distinto
    plt.scatter(puntos_cluster[:, 0], puntos_cluster[:, 2], color=colores[j], label=f'Cluster {j+1}', marker='o')
# Graficar las medias en color azul
plt.scatter(medias[:, 0], medias[:, 2], color='black', marker='*', s=200, label='Centroides') 


plt.figure(3)
plt.title('Gráfico de clusters con sus medias respecto a la longitud del sepalo y el ancho del petalo')
plt.xlabel('Eje X1')
plt.ylabel('Eje X2')
plt.grid(True)
for j in range(k):
    # Extraer los puntos correspondientes al cluster j
    puntos_cluster = x[indices[j]]
    
    # Graficar los puntos del cluster con un color distinto
    plt.scatter(puntos_cluster[:, 0], puntos_cluster[:, 3], color=colores[j], label=f'Cluster {j+1}', marker='o')
# Graficar las medias en color azul
plt.scatter(medias[:, 0], medias[:, 3], color='black', marker='*', s=200, label='Centroides') 


plt.figure(4)
plt.title('Gráfico de clusters con sus medias respecto al ancho del sepalo y la longitud del petalo')
plt.xlabel('Eje X1')
plt.ylabel('Eje X2')
plt.grid(True)
for j in range(k):
    # Extraer los puntos correspondientes al cluster j
    puntos_cluster = x[indices[j]]
    
    # Graficar los puntos del cluster con un color distinto
    plt.scatter(puntos_cluster[:, 1], puntos_cluster[:, 2], color=colores[j], label=f'Cluster {j+1}', marker='o')
# Graficar las medias en color azul
plt.scatter(medias[:, 1], medias[:, 2], color='black', marker='*', s=200, label='Centroides') 


plt.figure(5)
plt.title('Gráfico de clusters con sus medias respecto al ancho del sepalo y el ancho del petalo')
plt.xlabel('Eje X1')
plt.ylabel('Eje X2')
plt.grid(True)
for j in range(k):
    # Extraer los puntos correspondientes al cluster j
    puntos_cluster = x[indices[j]]
    
    # Graficar los puntos del cluster con un color distinto
    plt.scatter(puntos_cluster[:, 1], puntos_cluster[:, 3], color=colores[j], label=f'Cluster {j+1}', marker='o')
# Graficar las medias en color azul
plt.scatter(medias[:, 1], medias[:, 3], color='black', marker='*', s=200, label='Centroides') 


plt.figure(6)
plt.title('Gráfico de clusters con sus medias')
plt.xlabel('Eje X1')
plt.ylabel('Eje X2')
plt.grid(True)
for j in range(k):
    # Extraer los puntos correspondientes al cluster j
    puntos_cluster = x[indices[j]]
    
    # Graficar los puntos del cluster con un color distinto
    plt.scatter(puntos_cluster[:, 2], puntos_cluster[:, 3], color=colores[j], label=f'Cluster {j+1}', marker='o')
# Graficar las medias en color azul
plt.scatter(medias[:, 2], medias[:, 3], color='black', marker='*', s=200, label='Centroides') 

plt.legend()
plt.show()