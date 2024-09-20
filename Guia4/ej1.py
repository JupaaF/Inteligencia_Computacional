import numpy as np
import matplotlib.pyplot as plt

from radio import radio
from radio_2 import actualizar_radio
from graficar_vecinos import graficar_vecinos

def indice_minimo(matrix):
    # Encontrar el índice del elemento mínimo en la matriz "aplanada"
    index_min = np.argmin(matrix)
    
    # Convertir el índice del vector aplanado a coordenadas (i, j)
    indices = np.unravel_index(index_min, matrix.shape)
    
    return indices

#leer archivos: -------------------
x = np.loadtxt('Guia4\circulo.csv', delimiter=',')
[cant_filas,cant_columnas]= x.shape

#Parámetros --------------------
N1 = 7
N2 = 7

pesos = np.empty((N1,N2), object)
dif_aux = np.empty((N1,N2), object)
distancias = np.empty((N1,N2), object)

for i in range(N1):
    for j in range(N2):
        pesos[i,j] = np.random.uniform(-0.5,0.5,(2,))


#### ENTRENAMIENTO ---------
cant_epocas = 500
contador = 0
# ---------- ORDENAMIENTO GLOBAL ----------
it = 0
r = int(N2/2)
v = 0.9

matriz_ganadores = np.empty((N1,N2),int)

# Inicializamos la ventana de la gráfica
# plt.ion()  # Modo interactivo para actualizar la gráfica
# fig = plt.figure()

while(it < cant_epocas):
    it+=1
    for p in range(cant_filas): # --------Patrones
        for i in range (N1): # ------Filas matriz pesos
            for j in range (N2): # ------Filas matriz pesos
                dif = x[p]-pesos[i,j]
                #(o implementar guardando el min, con sus indices)
                distancias[i,j]= np.dot(dif,dif.T) # calculamos la distancia euclidea pero sin hacer la raiz
                # print(distancias[i,j])
        
        
        #Seleccion de la neurona ganadora
        fila_ganadora, columna_ganadora = indice_minimo(distancias)
        
        #Adaptacion de los pesos
        dif_aux = x[p]-pesos[fila_ganadora,columna_ganadora]

        # matriz_ganadores[fila_ganadora,columna_ganadora] += 1
        delta_w = v*dif_aux

        #### maximizar el ajuste del delta_w
        # max_adjustment = 0.1  # Define un límite para los ajustes
        # if np.linalg.norm(delta_w) > max_adjustment:
        #     delta_w = max_adjustment * (delta_w / np.linalg.norm(delta_w))

        # print(delta_w)
        radio(r, pesos, fila_ganadora, columna_ganadora,delta_w, N1, N2, 0)

    if(it == 100):
        print(pesos)
        exit()
    
    # if(contador > cant_epocas*0.1):
    #     contador = 0
    #     graficar_vecinos(pesos)
    # else:
    #     contador+=1

### print de prueba
# print(matriz_ganadores)

# for i in range (N1):
#     for j in range (N2):
#         print(f'Fila: {i} Columna: {j} Peso: {pesos[i,j]}')


#### terminar de graficar
# plt.ioff()  # Desactivar modo interactivo
# plt.show()  # Mostrar la última imagen
    

##### -------- TRANSICIÓN ----------
# it = 0
# r_cte = r -1
# v_cte = v- 0.1
# cant_epocas = 1000

# while(it<cant_epocas):
    
#     r = int(r_cte*(cant_epocas - it)/cant_epocas) + 1

#     v = (v_cte*(cant_epocas - it)/cant_epocas) + 0.1


#     it+=1
#     for p in range(cant_filas): # --------Patrones
#         for i in range (N1): # ------Filas matriz pesos
#             for j in range (N2): # ------Filas matriz pesos
#                 dif = x[p]-pesos[i,j]
#                 #(o implementar guardando el min, con sus indices)
#                 distancias[i,j]= np.dot(dif,dif.T) # calculamos la distancia euclidea pero sin hacer la raiz
       
#         #Seleccion de la neurona ganadora
#         fila_ganadora, columna_ganadora = indice_minimo(distancias)

#         #Adaptacion de los pesos
#         dif_aux = x[p]-pesos[fila_ganadora,columna_ganadora]
#         delta_w = v*dif_aux
#         radio(r, pesos, fila_ganadora, columna_ganadora,delta_w, N1, N2, 0)

# for i in range (N1):
#     for j in range (N2):
#         print(f'Fila: {i} Columna: {j} Peso: {pesos[i,j]}')
# ##### -------- CONVERGENCIA ----------
# it = 0
# r = 0
# v = 0.1
# cant_epocas = 2000

# while(it<cant_epocas):
    
#     it+=1
#     for p in range(cant_filas): # --------Patrones
#         for i in range (N1): # ------Filas matriz pesos
#             for j in range (N2): # ------Filas matriz pesos
#                 dif = x[p]-pesos[i,j]
#                 #(o implementar guardando el min, con sus indices)
#                 distancias[i,j]= np.dot(dif,dif.T) # calculamos la distancia euclidea pero sin hacer la raiz
       
#         #Seleccion de la neurona ganadora
#         fila_ganadora, columna_ganadora = indice_minimo(distancias)

#         #Adaptacion de los pesos
#         dif_aux = x[p]-pesos[fila_ganadora,columna_ganadora]
#         delta_w = v*dif_aux
#         radio(r, pesos, fila_ganadora, columna_ganadora,delta_w, N1, N2, 0)










################

    # if(contador > cant_epocas*0.01):
    #     contador = 0
    #     graficar_vecinos(pesos)
    # else:
    #     contador+=1



# plt.ioff()  # Desactivar modo interactivo
# plt.show()  # Mostrar la última imagen








            














# delta_w = np.array([1,1])

# for i in range (N1):
#     for j in range (N2):
#         print(f'Fila: {i} Columna: {j} Peso: {pesos[i,j]}')
    
# radio(3, pesos,2,3,delta_w,N1,N2,0)

# for i in range (N1):
#     for j in range (N2):
#         print(f'Fila: {i} Columna: {j} Peso: {pesos[i,j]}')









