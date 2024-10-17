import numpy as np
import random 
import sklearn 
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn import svm

cant_bits = 7129

cant_individuos = 30
paridad = (cant_individuos+1)%2 
cant_iteraciones = 50
it = 0
tasa_mutacion_individuo = 10
cant_media_columnas = 100
coef_ACC = 10
coef_col = 5


#DATOS
#entrenamiento
train = np.loadtxt('Guía6/leukemia_train.csv', delimiter=',')
x_train = train[:,:-1]
y_train = train[:,-1]
#prueba
test = np.loadtxt('Guía6/leukemia_test.csv', delimiter=',')
x_test = test[:,:-1]
y_test = test[:,-1]

def gen_fen(x):
    indices = []
    for i in range(len(x)):
        if(x[i]==1):
            indices.append(int(i))
    return indices

def fitness(x):
    
    ## SVM polinomial 
    clf = svm.SVC(kernel='poly',degree=2,coef0=1,gamma='auto')
    x_aux_train = x_train[:,x]
    x_aux_test = x_test[:,x]

    # clf = MLPClassifier(hidden_layer_sizes=(12,6),learning_rate_init=0.005,max_iter=300,activation='logistic',early_stopping=True,validation_fraction=0.3,shuffle=True,random_state=0)
    clf.fit(x_aux_train,y_train)
    y_pred = clf.predict(x_aux_test)
    ACC = accuracy_score(y_test,y_pred) ## calcula la tasa de precisión
    
    f = (coef_ACC*ACC)/(coef_col*len(x))
    return f

def ordenar(vector_f):
    indices = list(range(len(vector_f)))
    indices_ordenados = sorted(indices, key=lambda i:vector_f[i],reverse=True)
    return indices_ordenados

#Inicializar población --> 1) crear individuos como cadenas de bits
poblacion = np.empty((cant_individuos,cant_bits),int)

for i in range(cant_individuos):
    for j in range(cant_bits):
        if(random.randint(0,cant_bits)<cant_media_columnas):
            poblacion[i,j] = 1
        else:
            poblacion[i,j] = 0

#Evaluar-->1) traducir cadena de bit a parametros    
vector_fen = np.empty(cant_individuos,object)
for i in range(cant_individuos):
    vector_fen[i] = gen_fen(poblacion[i,:])


# -->2) evaluar y guardar valores fitness
f = np.empty(cant_individuos,float)
for i in range(cant_individuos):
    f[i] = fitness(vector_fen[i])
indices_ord = ordenar(f)

#Repetir hasta cumplir aptitud
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
    f = np.empty(cant_individuos,float)
    for i in range(cant_individuos):
        f[i] = fitness(vector_fen[i])
    indices_ord = ordenar(f)
    
    it += 1
    print(it)

resultado = gen_fen(poblacion[indices_ord[0],:])
print(((best_fitness[-1]*coef_col)*len(resultado))/coef_ACC)
print(resultado)
print(len(resultado))


plt.plot(best_fitness)
plt.title('Evolución del mejor fitness')
plt.xlabel('Iteración')
plt.ylabel('Mejor Fitness')
plt.grid(True)
plt.show()