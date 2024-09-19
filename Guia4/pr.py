import numpy as np
import matplotlib.pyplot as plt

#leer archivos: -------------------
x = np.loadtxt('Guia4\circulo.csv', delimiter=',')
pesos = np.random.uniform(-0.5,0.5,(2,))


dif = x[0]-pesos
print(x[0])
print(pesos)
print(dif)

d= np.dot(dif,dif.T)
print(d)