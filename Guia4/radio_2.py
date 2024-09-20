import numpy as np

def actualizar_radio(r, pesos, fila_ganadora, col_ganadora, delta_w, N1, N2, decay_factor=0.5):
    # """
    # Actualiza los pesos de la neurona ganadora y sus vecinas dentro del radio dado.
    
    # Parámetros:
    # r            : radio de influencia (int).
    # pesos        : matriz de pesos (np.array).
    # fila_ganadora: fila de la neurona ganadora (int).
    # col_ganadora : columna de la neurona ganadora (int).
    # delta_w      : ajuste de los pesos (np.array).
    # N1           : tamaño de la matriz de pesos en filas (int).
    # N2           : tamaño de la matriz de pesos en columnas (int).
    # decay_factor : factor de reducción del ajuste con la distancia (float, default=0.5).
    # """
    
    # Recorrer todas las neuronas en el rango del radio
    for i in range(-r, r+1):  # Movimiento en las filas
        for j in range(-r, r+1):  # Movimiento en las columnas
            fila_vecina = fila_ganadora + i
            col_vecina = col_ganadora + j
            
            # Asegurarse de que la neurona vecina está dentro de los límites de la matriz
            if 0 <= fila_vecina < N1 and 0 <= col_vecina < N2:
                # Calcular la distancia de la neurona vecina a la neurona ganadora
                distancia = np.sqrt(i**2 + j**2)
                
                # Aplicar un factor de reducción basado en la distancia (más lejos, menos ajuste)
                ajuste = delta_w * np.exp(-distancia * decay_factor)
                
                # Actualizar los pesos de la neurona vecina
                pesos[fila_vecina, col_vecina] += ajuste

