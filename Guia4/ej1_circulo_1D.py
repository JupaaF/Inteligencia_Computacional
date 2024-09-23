import numpy as np
import matplotlib.pyplot as plt
from funcion_modif import funcion_modif_segun_entorno
from graficar_vecinos import graficar_vecinos

#leer archivos: -------------------
x = np.loadtxt('Guia4/circulo.csv', delimiter=',')
[cant_filas,cant_columnas]= x.shape

#Parmetros del SOM ----------
N1 = 7#(4x4)
N2 = 1
cantidad_entradas = 2

#Inicializamos pesos
pesos = np.empty((N1,N2), object)

for i in range(N1):
    for j in range(N2):
        pesos[i,j] = np.random.uniform(-0.5,0.5,(2,))

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

