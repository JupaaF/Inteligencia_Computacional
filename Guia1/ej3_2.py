# ejercicio 1 -  código de entrenamiento
import numpy as np
import random
import matplotlib.pyplot as plt

def graficar(w_r):
    pendiente=-(w_r[1]/w_r[2])
    ordenada = w_r[0]/w_r[2]

    x1 = np.linspace(-1.1, 1.1, 1000, endpoint=False)
    x2 = ordenada + pendiente*x1
    plt.clf()
    plt.title('Gráfico de Líneas')
    plt.xlabel('Eje X1')
    plt.ylabel('Eje X2')
    plt.axhline(0, color='black', linewidth=1)  # Línea horizontal en y=0
    plt.axvline(0, color='black', linewidth=1)  # Línea vertical en x=0
    plt.plot(x1, x2, marker='.', linestyle='-', linewidth=0.01)
    plt.scatter(x[:,1], x[:,2], color='g', marker='.')    
    plt.ylim(-1.5,1.5)
    plt.xlim(-1.5,1.5)
    plt.pause(0.1)
#leer archivo:
#entrenamiento
x = np.loadtxt('archivos/OR_90_trn.csv', delimiter=',')
#prueba
p = np.loadtxt('archivos/OR_90_tst.csv', delimiter=',')

# PROBAMOS LA PRUEBA CON EL DE 50 tambien :D

[filas,columnas]= x.shape
[filasp,columnasp]= p.shape

#agregar columna al principo
x0= np.full(filas, -1).reshape(-1, 1)
p0= np.full(filasp, -1).reshape(-1, 1)
x = np.hstack((x0, x))
p = np.hstack((p0, p))
columnas = columnas+1

#separo los y deseados
yd = x[:,-1]
yd_p = p[:,-1]
 
#generar pesos random
w_r = np.zeros(columnas-1) 
for i in range(w_r.size):
    w_r[i] = np.random.uniform(0.0, 1.0)-0.5


#plot
# Añadir título y etiquetas

plt.grid(True)
plt.ion()


#tasa de aprendizaje
v = 0.01
#tasa de desempeño
d1 = 0
d2 = 0
#ITERACIONES ENTRENAMIENTO
it = 0

#bucle para entrenar
while  d1/filas < 0.95 and it <10:
    d1=0
    it+=1
    for i in range(filas):
        y = np.dot(w_r,x[i,:columnas-1]) #calculamos y
        for j in range(columnas-1):
            w_r[j] = w_r[j]+ v*(yd[i]-y)*x[i,j] #corregimos pesos
        if(i%50 == 1):
            graficar(w_r)
  
    for t in range(filas):
        y = np.dot(w_r,x[t,:columnas-1]) #calculamos y
        if(yd[t]== np.sign(y)): d1+=1

#prueba
for i in range(filasp):
    y = np.dot(w_r,p[i,:columnas-1]) #calculamos y
    if(yd_p[i]== np.sign(y)): d2+=1      

print(f'Desempeño Entrenamiento: {100*d1/filas} ')
print(f'Desempeño Prueba: {100*d2/filasp} ')

