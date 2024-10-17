import numpy as np
import random 
import matplotlib.pyplot as plt
import math

cant_particulas = 50
dimensiones = 2
x1 = -100
x2 = 100
c1 = 1.5 #cte de aceleracion 1
c2 = 0.5 #cte de aceleracion 2
rango = np.abs(c1-c2) 
cant_iteraciones = 200
it = 0

def fitness(x,y):
    f = ((x**2 + y**2)**0.25) * (math.sin(50 * ((x**2 + y**2)**0.1))**2 + 1)
    return f

#inicializar particulas y vector de mejores individuales
x = np.empty((cant_particulas,dimensiones),float)
v = np.zeros((cant_particulas,dimensiones))
# v = np.empty((cant_particulas,dimensiones),float)
y = np.empty((cant_particulas,dimensiones),float)
mejores_valores = []

# for i in range(cant_particulas):
#     for j in range(dimensiones):
#         v[i,j] = random.uniform(0,5)

for i in range(cant_particulas):
    for j in range(dimensiones):
        y[i,j] = random.uniform(x1,x2)

for i in range(cant_particulas):
    for j in range(dimensiones):
        x[i,j] = random.uniform(x1,x2)
       # y[i,j] = x[i,j]

f_x = np.empty(cant_particulas,float)
f_y = np.empty(cant_particulas,float)
for i in range(cant_particulas):
    f_x[i] = fitness(x[i,0],x[i,1])
    f_y[i] = fitness(y[i,0],y[i,1])

#------------------
index_min_y = np.argmin(f_y)
f_global = f_y[index_min_y]
xy_global = y[index_min_y]

while(it<cant_iteraciones):
    #para cada particula
    for i in range(cant_particulas):
        if(f_x[i]<f_y[i]):
            y[i] = x[i]
        if(f_y[i]<f_global):
            xy_global = y[i]
            f_global = f_y[i]

    for i in range(cant_particulas):
        for j in range(dimensiones): # --> dimensiones de la particula
            r1 = random.uniform(0,1)
            r2 = random.uniform(0,1)
            #actualizamos velocidad
            v[i,j] = v[i,j] + c1*r1*(y[i,j]-x[i,j]) + c2*r2*(xy_global[j]-x[i,j])

        #actualizamos posicion
        x[i] = x[i]+v[i]
        if((x[i,0]>x2 or x[i,0]<x1)or(x[i,1]>x2 or x[i,1]<x1)):
            x[i] = x[i]-v[i]

    for i in range(cant_particulas):
        f_x[i] = fitness(x[i,0],x[i,1])
        f_y[i] = fitness(y[i,0],y[i,1])
    
    print(it)
    
    it+=1
    c1 = c1+(1/cant_iteraciones)*rango
    c2 = c2-(1/cant_iteraciones)*rango

    mejores_valores.append(f_global)


index_min_y = np.argmin(f_y)
f_global = f_y[index_min_y]
xy_global = y[index_min_y]

print(f"Mejor particula encontrada: x:{xy_global[0]}, y: {xy_global[1]}")
print(f"Valor de y: {f_global}")

plt.plot(mejores_valores)
plt.xlabel("Iteraciones")
plt.ylabel("Mejor valor encontrado")
plt.show()