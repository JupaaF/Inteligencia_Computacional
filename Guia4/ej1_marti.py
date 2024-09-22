import numpy as np
import matplotlib.pyplot as plt
from radio import radio
from radio_2 import actualizar_radio
from graficar_vecinos import graficar_vecinos

def distancia_sin_raiz(vector1, vector2):
    # Convertir los vectores en arrays de NumPy
    v1 = np.array(vector1)
    v2 = np.array(vector2)
    
    # Calcular la suma de los cuadrados de las diferencias
    suma_cuadrados = np.sum((v1 - v2) ** 2)
    
    return suma_cuadrados


#leer archivos: -------------------
x = np.loadtxt('Guia4/circulo.csv', delimiter=',')
[cant_filas,cant_columnas]= x.shape

#Parmetros del SOM ----------
N1 = 7#(4x4)
N2 = 7
cantidad_entradas = 2


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
cant_epocas = 900
it = 0
r = 3
v = 0.9
i_ganadora = 0
j_ganadora = 0
contador = 0
g = 2 # graficar cada g iteraciones

while(it<cant_epocas):
    it+=1
    # print(it)
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
        print(modif_pesos)
        
        #Recorrer pesos para modificar los que esten dentro del entorno de la i,j ganadora
        for i in range(N1):
            for j in range(N2):
                # Calcular la distancia de Manhattan desde (i_ganadora, j_ganadora) a (i, j)
                distancia = abs(i - i_ganadora) + abs(j - j_ganadora) 
                
                # Si la distancia es menor o igual a R, modificar peso
                if(distancia <= r):
                    pesos[i, j] = pesos[i, j] + modif_pesos
                
    if(it % g == 0):
        graficar_vecinos(pesos)
    
plt.ioff()  # Desactivar modo interactivo
plt.show()  # Mostrar la última imagen

