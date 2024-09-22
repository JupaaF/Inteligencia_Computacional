import numpy as np
import matplotlib.pyplot as plt

from radio import radio
from graficar_vecinos import graficar_vecinos


# leer archivos: -------------------
x = np.loadtxt('Guia4/circulo.csv', delimiter=',')
[cant_filas, cant_columnas] = x.shape

# Parámetros --------------------
N1 = 6
N2 = 7

pesos = np.empty((N1, N2), object)
dif_aux = np.empty((N1, N2), object)

for i in range(N1):
    for j in range(N2):
        pesos[i, j] = np.random.uniform(-0.5, 0.5, (2,))

#### ENTRENAMIENTO ---------
cant_epocas = 500
contador = 0
# ---------- ORDENAMIENTO GLOBAL ----------
it = 0
r = int(N2 / 2)  # Radio inicial
v = 0.7  # Tasa de aprendizaje inicial

# Inicializamos la ventana de la gráfica
plt.ion()  # Modo interactivo para actualizar la gráfica
fig = plt.figure()

while it < cant_epocas:
    it += 1
    for p in range(cant_filas):  # --------Patrones
        distancia_ganadora = 1000
        for i in range(N1):  # ------Filas matriz pesos
            for j in range(N2):  # ------Columnas matriz pesos
                dif = x[p] - pesos[i, j]
                distancia = np.dot(dif, dif.T)  # Calculamos la distancia euclidea sin la raíz

                if distancia < distancia_ganadora:
                    fila_ganadora = i
                    columna_ganadora = j
                    distancia_ganadora = distancia
                    dif_ganadora = dif

        # Actualización de los pesos
        delta_w = v * dif_ganadora

        #### Maximizar el ajuste del delta_w
        max_adjustment = 0.05  # Límite más bajo
        if np.linalg.norm(delta_w) > max_adjustment:
            delta_w = max_adjustment * (delta_w / np.linalg.norm(delta_w))

        # Actualizar los pesos de la neurona ganadora y sus vecinos
        radio(r, pesos, fila_ganadora, columna_ganadora, delta_w, N1, N2, 0)

    # Reducir la tasa de aprendizaje y el radio con el tiempo
    v = 0.7 * np.exp(-it / cant_epocas)  # Decremento exponencial de la tasa de aprendizaje
    r = int((N2 / 2) * np.exp(-it / cant_epocas))  # Decremento del radio

    if contador > cant_epocas * 0.05:
        contador = 0
        graficar_vecinos(pesos)
    else:
        contador += 1

#### terminar de graficar
plt.ioff()  # Desactivar modo interactivo
plt.show()  # Mostrar la última imagen
