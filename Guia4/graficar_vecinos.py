import numpy as np
import matplotlib.pyplot as plt

def graficar_vecinos(pesos):
    N1, N2 = pesos.shape[:2]
    
    # Limpiar la gráfica anterior para actualizar en la misma ventana
    plt.cla()

    for i in range(N1):
        for j in range(N2):
            # Obtener el peso actual como un vector 2D
            current_point = pesos[i, j]

            # Conectar con el vecino de la derecha (si existe)
            if j + 1 < N2:
                right_neighbor = pesos[i, j + 1]
                plt.plot([current_point[0], right_neighbor[0]], 
                         [current_point[1], right_neighbor[1]], 'bo-')  # Azul

            # Conectar con el vecino de abajo (si existe)
            if i + 1 < N1:
                bottom_neighbor = pesos[i + 1, j]
                plt.plot([current_point[0], bottom_neighbor[0]], 
                         [current_point[1], bottom_neighbor[1]], 'bo-')  # Rojo

    # Etiquetas y título opcionales
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Conexión de vecinos inmediatos en la matriz de pesos')
    plt.grid(True)

    # Pausar para visualizar la actualización en vivo
    plt.pause(0.1)



