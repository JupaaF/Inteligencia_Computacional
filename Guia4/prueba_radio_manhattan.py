import numpy as np

def sumar_radio_manhattan(matriz, i, j, R):
    # Obtener las dimensiones de la matriz
    filas, columnas = matriz.shape
    
    # Recorrer todas las posiciones dentro de la matriz
    for x in range(filas):
        for y in range(columnas):
            # Calcular la distancia de Manhattan desde (i, j) a (x, y)
            distancia = abs(x - i) + abs(y - j)
            
            # Si la distancia es menor o igual a R, sumar 1
            if distancia <= R:
                matriz[x, y] += 1

# Ejemplo de uso
matriz = np.zeros((5, 5))  # Crear una matriz de 5x5 con todos los valores en 1
i, j = 4,4              # Índice central
R = 2                     # Radio

sumar_radio_manhattan(matriz, i, j, R)
print(matriz)

