import numpy as np
import matplotlib.pyplot as plt

### prueba minimo
# def indice_minimo(matrix):
#     # Encontrar el índice del elemento mínimo en la matriz "aplanada"
#     index_min = np.argmin(matrix)
    
#     # Convertir el índice del vector aplanado a coordenadas (i, j)
#     indices = np.unravel_index(index_min, matrix.shape)
    
#     return indices

# size = 6
# num_elements = size * size
# random_elements = np.random.permutation(num_elements) + 1  # Generar números del 1 al 36 en orden aleatorio
# matrix = random_elements.reshape(size, size)
# print(matrix)

# fila, columna= indice_minimo(matrix)
# print(fila,columna)





#leer archivos: -------------------
# x = np.loadtxt('Guia4\circulo.csv', delimiter=',')
# pesos = np.random.uniform(-0.5,0.5,(2,))


# dif = x[0]-pesos
# print(x[0])
# print(pesos)
# print(dif)

# d= np.dot(dif,dif.T)
# print(d)

