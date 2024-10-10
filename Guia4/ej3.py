import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn import datasets

x = np.loadtxt('Guia4/irisbin_trn.csv', delimiter=',')
[cant_filas,cant_columnas]= x.shape
x = x[:,:-3]
davies_bouldin_scores = []
silhouette_scores = []
inertia_values = []

for k in range(2,11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(x)
    
    # Predecir las etiquetas de los clústeres
    clusters = kmeans.labels_

    #Separación: -------------------
    # Mide qué tan cerca están los puntos de los clústeres vecinos. 
    # --> Los valores varían de -1 a 1, 
    #     donde un valor cercano a 1 indica que los puntos están bien agrupados.
    silhouette = silhouette_score(x, clusters)
    silhouette_scores.append(silhouette)

    #Coherencia: -------------------
    # Es la suma de las distancias al cuadrado entre los puntos y el centro de su clúster. 
    # --> Cuanto menor sea la inercia, mayor será la cohesión de los clústeres, 
    #     ya que indica que los puntos están más cerca de sus centroides.
    inertia = kmeans.inertia_
    inertia_values.append(inertia)

    #Davies Bouldin: ------------------- 
    # Usa la separación y la coherencia.
    # --> Valores más bajos son mejores, indicando que los clústeres son más compactos y separados.
    davies_bouldin = davies_bouldin_score(x, clusters)
    davies_bouldin_scores.append(davies_bouldin)

k = np.linspace(2,10,9)

# Graficar los resultados
plt.figure(figsize=(14, 5))

# Silhouette Score
plt.subplot(1, 3, 1)
plt.grid()
plt.plot(k, silhouette_scores, marker='o')
plt.title('Silhouette Score por k')
plt.xlabel('Número de Clústeres (k)')
plt.ylabel('Separación')

# Davies-Bouldin Score
plt.subplot(1, 3, 3)
plt.grid()
plt.plot(k, davies_bouldin_scores, marker='o', color='r')
plt.title('Davies-Bouldin Score por k')
plt.xlabel('Número de Clústeres (k)')
plt.ylabel('Davies-Bouldin')

# Inertia
plt.subplot(1, 3, 2)
plt.grid()
plt.plot(k, inertia_values, marker='o',color='g')
plt.title('Inertia por k')
plt.xlabel('Número de Clústeres (k)')
plt.ylabel('Cohesión')

plt.tight_layout()
plt.show()