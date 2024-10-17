import numpy as np
import random 
import matplotlib.pyplot as plt
import math
import autograd.numpy as np
from autograd import grad
cant_bits = 40 #primeros 20 para x, segundos 20 para y

x1 = -100
x2 = 100
cant_individuos = 25
paridad = (cant_individuos+1)%2 
cant_iteraciones = 5000
it = 0
tasa_mutacion_individuo = 15


def gen_fen(x):
    sum_x = 0
    for i in range (20):
        sum_x += (2**(20-i-1))*x[i]
    
    sum_x = x1 + sum_x * (x2 - x1) / (2**20 - 1)
    sum_y = 0
    for i in range (20,40):
        sum_y += (2**(40-i-1))*x[i]
    
    sum_y = x1 + sum_y * (x2 - x1) / (2**20 - 1)
    return [sum_x,sum_y]

def fitness(xy):
    y = ((xy[0]**2 + xy[1]**2)**0.25) * (math.sin(50 * ((xy[0]**2 + xy[1]**2)**0.1))**2 + 1)
    return -y

def ordenar(vector_f):
    indices = list(range(len(vector_f)))
    indices_ordenados = sorted(indices, key=lambda i:vector_f[i],reverse=True)
    return indices_ordenados

#Inicializar población --> 1) crear individuos como cadenas de bits
poblacion = np.empty((cant_individuos,cant_bits),int)

for i in range(cant_individuos):
    for j in range(cant_bits):
        poblacion[i,j] = random.randint(0,1)

#Evaluar-->1) traducir cadena de bit a parametros    
vector_fen = np.empty((cant_individuos,2),float)
for i in range(cant_individuos):
    vector_fen[i] = gen_fen(poblacion[i,:])

# -->2) evaluar y guardar valores fitness
f = np.empty(cant_individuos,float)
for i in range(cant_individuos):
    f[i] = fitness(vector_fen[i])
indices_ord = ordenar(f)
# #Repetir hasta cumplir aptitud
best_fitness = []

while(it < cant_iteraciones):
 #Generar nueva poblacion
    ## definir hijitos y papas
    hijos = np.empty((cant_individuos,cant_bits),int)
    padres = np.empty((cant_individuos,cant_bits),int)
    hijos[paridad] = poblacion[indices_ord[paridad]]
    hijos[0] = poblacion[indices_ord[0]]
    best_fitness.append(f[indices_ord[0]])

    #seleccionar padres (metodo de ventana)
    for i in range(cant_individuos-1-paridad): #cantidad de ventanas 
           #a) ruelta:  f = [4.5 2 3 0.5] divido sobre la suma
            #          =[0.45 0.2 0.3 0.05]
            # => hago sumo hasta cada uno: [0.45 0.6 0.95 1]
            #      tiro un nro random entre 0 y 1 y si es 0<x<0.45, elijo el primero, etc etc
            #b) ventana:  ordeno por fitness, elijo ventana, saco uno aleat de ese rango
            #              y repito para otra ventana pequeña (puede salir 2 o más veces el mismo individio)   
            #c) competencia:
        padres[i] = poblacion[indices_ord[random.randint(0,cant_individuos-i-1)],:]

    #cruzas
    for i in range(0,cant_individuos-2,2):
        punto_de_cruce = random.randint(1,cant_bits-1)
        hijos[i+1+paridad] = np.concatenate((padres[i,0:punto_de_cruce], padres[i+1,punto_de_cruce:]))
        hijos[i+2+paridad] = np.concatenate((padres[i+1,0:punto_de_cruce], padres[i,punto_de_cruce:])) 


    #reemplazamos toda la poblacion
    poblacion = hijos
    #mutacion a todos los indiv (probabilidad muy baja)
    for i in range(1,cant_individuos):
        if(random.randint(0,99)< tasa_mutacion_individuo):
            indice_mutacion = random.randint(0,cant_bits-1)
            poblacion[i, indice_mutacion] = 1 - poblacion[i, indice_mutacion]
    
    #evaluar fitness y guardar valores
    for i in range(cant_individuos):
        vector_fen[i] = gen_fen(poblacion[i,:])

    # -->2) evaluar y guardar valores fitness
    for i in range(cant_individuos):
        f[i] = fitness(vector_fen[i])
    indices_ord = ordenar(f)
    
    it += 1
    print(it)



resultado = gen_fen(poblacion[indices_ord[0],:])

print(f"Algoritmo Genético: --------------")
print(f"El valor mínimo de x,y es: {[float(resultado[0]),float(resultado[1])]}")
print(f"El valor mínimo de la función es: {-best_fitness[-1]}")

# Metodo del gradiente descendiente -------------------------------------

# Definir la función objetivo
def f(xy):
    x, y = xy
    r = (x**2 + y**2)
    return r**0.25 * (np.sin(50 * r**0.1)**2 + 1)

# Gradiente automático con Autograd
gradient_f = grad(f)

# Gradiente descendente usando el gradiente de Autograd
def gradient_descent_2d_autograd(x0, y0, learning_rate, tolerance, max_iter=10000):
    xy = np.array([x0, y0])
    for i in range(max_iter):
        grad_xy = gradient_f(xy)
        if np.linalg.norm(grad_xy) < tolerance:
            break
        xy -= learning_rate * grad_xy
        if((xy[0]>100 or xy[0]<-100)or(xy[1]>100 or xy[1]<-100)):
            xy += learning_rate * grad_xy
    return xy[0], xy[1]

# Parámetros iniciales
x0, y0 = np.random.uniform(-100, 100, 2)  # Puntos iniciales aleatorios
learning_rate = 0.01                      # Tasa de aprendizaje
tolerance = 1e-6                         # Tolerancia para detener

# Ejecutar gradiente descendente
min_x, min_y = gradient_descent_2d_autograd(x0, y0, learning_rate, tolerance)

print(f"Metodo del gradiente descendiente: --------------")
print(f"El valor mínimo de x es: {min_x}")
print(f"El valor mínimo de y es: {min_y}")
print(f"El valor mínimo de la función es: {f([min_x, min_y])}")










# ----------

plt.plot(best_fitness)
plt.title('Evolución del mejor fitness')
plt.xlabel('Iteración')
plt.ylabel('Mejor Fitness')
plt.grid(True)
plt.show()