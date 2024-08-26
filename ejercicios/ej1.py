# ejercicio 1 -  código de entrenamiento
import numpy as np
import random

#leer archivo:
#entrenamiento
x = np.loadtxt('C:\\Users\\piliv\\OneDrive\\Documentos\\FACU\\Inteligencia Computacional\\Trabajos prácticos\\Guia1\\OR_trn.csv', delimiter=',')
#prueba
p = np.loadtxt('C:\\Users\\piliv\\OneDrive\\Documentos\\FACU\\Inteligencia Computacional\\Trabajos prácticos\\Guia1\\OR_tst.csv', delimiter=',')
[filas,columnas]= x.shape
[filasp,columnasp]= p.shape

#agregar columna de pesos w0 al principo
x0= np.full(filas, -1).reshape(-1, 1)
p0= np.full(filasp, -1).reshape(-1, 1)
x = np.hstack((x0, x))
p = np.hstack((p0, p))
columnas = columnas+1

#separo las salidas deseadas
yd = x[:,-1]
yd_p = p[:,-1]
 
#generar pesos aleatorios para comenzar
w_r = np.zeros(columnas-1) 
for i in range(w_r.size):
    w_r[i] = np.random.uniform(0.0, 1.0)-0.5

#tasa de aprendizaje
v = 0.5
#tasa de desempeño
d1 = 0
d2 = 0
#ITERACIONES ENTRENAMIENTO
it = 0

#bucle para entrenar
while  d1/filas < 0.99 and it <500:
    d1=0
    it+=1
    for i in range(filas):
        y = np.dot(w_r,x[i,:columnas-1]) #calculamos y
        for j in range(columnas-1):
            w_r[j] = w_r[j]+ v*(yd[i]-y)*x[i,j] #corregimos pesos
  
    for t in range(filas):
        y = np.dot(w_r,x[t,:columnas-1]) #calculamos y
        if(yd[t]== np.sign(y)): d1+=1

#prueba
for i in range(filasp):
    y = np.dot(w_r,p[i,:columnas-1]) #calculamos y
    if(yd_p[i]== np.sign(y)): d2+=1      

print(f'Desempeño Entrenamiento: {100*d1/filas} ')
print(f'Desempeño Prueba: {100*d2/filasp} ')