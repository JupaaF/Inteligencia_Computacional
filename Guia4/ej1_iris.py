import numpy as np
import matplotlib.pyplot as plt
from funcion_modif import funcion_modif_segun_entorno
from graficar_vecinos import graficar_vecinos
from generar_cluster_som import gen_cluster_iris
import random
#leer archivos: -------------------
x = np.loadtxt('Guia4/irisbin_trn.csv', delimiter=',')
[cant_filas,cant_columnas]= x.shape
x = x[:,:-3]
#Parmetros del SOM ----------
N1 = 7#(4x4)
N2 = 7
cantidad_entradas = 4

#Inicializamos pesos
pesos = np.empty((N1,N2), object)

for i in range(N1):
    for j in range(N2):
        pesos[i,j] = np.random.uniform(-0.5,0.5,(cantidad_entradas,))

# Inicializamos la ventana de la gráfica
plt.ion()  # Modo interactivo para actualizar la gráfica
fig = plt.figure()

#  -------------------------------------------
# |               ENTRENAMIENTO               |
#  -------------------------------------------

# -------------- ORDENAMIENTO GLOBAL --------------
cant_epocas = 15
it = 0
r = int(N1/2)
v = 0.7
i_ganadora = 0
j_ganadora = 0
contador = 0
g = 2 # graficar cada g iteraciones

print('Iteraciones ORDENAMIENTO GLOBAL')
while(it<cant_epocas):
    it+=1
    print(it)

    for p in range(cant_filas): # Para cada patron x: --------
        menor_d = 1000000
        #Recorro pesos para encontrar neurona ganadora
        for i in range(N1):
            for j in range(N2):
                #Calculo de distancia
                distancia = np.sum((x[p,:] - pesos[i,j]) ** 2) 
                #Neurona ganadora
                if(distancia<=menor_d):
                    menor_d = distancia
                    i_ganadora=i
                    j_ganadora=j
        
        #Modificación a sumar a los pesos del entorno:
        modif_pesos = v*(x[p,:]-pesos[i_ganadora,j_ganadora])
        xx = x[p,:]
        funcion_modif_segun_entorno(r, pesos, modif_pesos, i_ganadora,j_ganadora,N1,N2,v,xx)
             
    if(it % g == 0):
        graficar_vecinos(pesos)
    



# -------------- TRANSICION --------------
cant_epocas = 20
it = 0
r_cte = r -1
v_cte = v- 0.1
i_ganadora = 0
j_ganadora = 0
contador = 0
g = 2 # graficar cada g iteraciones

print('Iteraciones TRANSICION')
while(it<cant_epocas):
    it+=1
    print(it)
    r = int(r_cte*(cant_epocas - it)/cant_epocas) + 1
    v = (v_cte*(cant_epocas - it)/cant_epocas) + 0.1

    for p in range(cant_filas): # Para cada patron x: --------
        menor_d = 1000000
        #Recorro pesos para encontrar neurona ganadora
        for i in range(N1):
            for j in range(N2):
                #Calculo de distancia
                distancia = np.sum((x[p,:] - pesos[i,j]) ** 2) 
                #Neurona ganadora
                if(distancia<=menor_d):
                    menor_d = distancia
                    i_ganadora=i
                    j_ganadora=j
        
        #Modificación a sumar a los pesos del entorno:
        modif_pesos = v*(x[p,:]-pesos[i_ganadora,j_ganadora])
        xx = x[p,:]
        funcion_modif_segun_entorno(r, pesos, modif_pesos, i_ganadora,j_ganadora,N1,N2,v,xx)
             
    if(it % g == 0):
        graficar_vecinos(pesos)
    

# -------------- CONVERGENCIA --------------
cant_epocas = 30
it = 0
r = 0
v = 0.09
i_ganadora = 0
j_ganadora = 0
contador = 0
g = 2 # graficar cada g iteraciones

print('Iteraciones CONVERGENCIA')
while(it<cant_epocas):
    it+=1
    print(it)

    for p in range(cant_filas): # Para cada patron x: --------
        menor_d = 1000000
        #Recorro pesos para encontrar neurona ganadora
        for i in range(N1):
            for j in range(N2):
                #Calculo de distancia
                distancia = np.sum((x[p,:] - pesos[i,j]) ** 2) 
                #Neurona ganadora
                if(distancia<=menor_d):
                    menor_d = distancia
                    i_ganadora=i
                    j_ganadora=j
        
        #Modificación a sumar a los pesos del entorno:
        modif_pesos = v*(x[p,:]-pesos[i_ganadora,j_ganadora])
        xx = x[p,:]
        funcion_modif_segun_entorno(r, pesos, modif_pesos, i_ganadora,j_ganadora,N1,N2,v,xx)
             
    if(it % g == 0):
        graficar_vecinos(pesos)
    
plt.ioff()  # Desactivar modo interactivo
plt.show()  # Mostrar la última imagen

x = np.loadtxt('Guia4/irisbin_trn.csv', delimiter=',')
v_medias = gen_cluster_iris(pesos,x)
x= x[:,:-3]
indices = [[] for _ in range(3)]
for i in range (cant_filas):
        dist_min = 10000
        indice_min = 0
        for j in range(3):
            vec_dist = x[i]-v_medias[j]
            dist = np.dot(vec_dist,vec_dist.T)
            if(dist<dist_min):
                indice_min = j
                dist_min = dist
        indices[indice_min].append(i)

######----------------------------------------K MEDIAS------------------------------------------------

x = np.loadtxt('Guia2\irisbin_trn.csv', delimiter=',')
[cant_filas,cant_columnas]= x.shape
cant_max_it = 1000
k = 3
x = x[:,:-3]
cant_columnas = cant_columnas-3

indices_k = [[] for _ in range(k)]
for i in range (cant_filas):
    n = random.randrange(k)
    indices_k[n].append(i)

medias = np.empty((k,cant_columnas),float)
for j in range (k):
    suma = 0
    for i in indices_k[j]:
        suma += x[i]
    medias[j,:]= suma/len(indices_k[j])

it=0
while it<cant_max_it:
    it+=1
    indices_k = [[] for _ in range(k)]
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
        indices_k[indice_min].append(i)
    for j in range (k):
        suma = 0
        for i in indices_k[j]:
            suma += x[i]
        if(len(indices_k[j])!=0):
            medias[j,:]= suma/len(indices_k[j])
    aux = medias - medias_anterior

    todos_ceros = True
    for fila in aux:
        for elemento in fila:
            if elemento != 0:
                todos_ceros = False
    if(todos_ceros):
        break