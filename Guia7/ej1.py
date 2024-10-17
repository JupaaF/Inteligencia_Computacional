import numpy as np
import random 

cant_particulas = 1000
x1 = -512
x2 = 512
c1 = 0.1 #cte de aceleracion 1
c2 = 0.01 #cte de aceleracion 2
cant_iteraciones = 100
it = 0



def fitness(x):
    y = -x*np.sin(np.sqrt(np.abs(x)))
    return y

#inicializar particulas y vector de mejores individuales
x = np.empty(cant_particulas,float)
v = np.zeros(cant_particulas) # -----------> ????
y = np.empty(cant_particulas,float)
for i in range(len(x)):
    x[i] = random.uniform(x1,x2)
    y[i] = random.uniform(x1,x2)

f_x = fitness(x)
f_y = fitness(y)
index_min_y = np.argmin(f_y)
f_global = f_y[index_min_y]
y_global = y[index_min_y]

while(it<cant_iteraciones):
    #para cada particula
    for i in range(len(x)):
        if(f_x[i]<f_y[i]):
            y[i] = x[i]
        if(f_y[i]<f_global):
            y_global = y[i]
            f_global = f_y[i]

    for i in range(len(x)):
        r1 = random.uniform(0,1)
        r2 = random.uniform(0,1)
        #actualizamos velocidad
        v[i] = v[i] + c1*r1*(y[i]-x[i]) + c2*r2*(y_global-x[i])

        #actualizamos posicion
        x[i] = x[i]+v[i]
        if(x[i]>512 or x[i]<-512):
            x[i] = x[i]-v[i]
            
        

    f_x = fitness(x)
    f_y = fitness(y)
    print(it)
    it+=1


index_min_y = np.argmin(f_y)
f_global = f_y[index_min_y]
y_global = y[index_min_y]
print(f"Mejor particula encontrada: {y_global}")
print(f"Valor de y: {f_global}")