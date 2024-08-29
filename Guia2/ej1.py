import numpy as np
import math
def sigmoide(z,a):
    for i in range(len(z)):
        z[i] = (1-math.e**(-a*z[i]))/(1+math.e**(-a*z[i]))
    return z

#leer archivos: ----------------
#entrenamiento
data_TR = np.loadtxt('archivos/OR_trn.csv', delimiter=',')
#prueba
data_TE = np.loadtxt('archivos/OR_tst.csv', delimiter=',')
[cant_filas,cant_columnas]= data_TR.shape
[cant_filas_p,cant_columnas_p]= data_TE.shape


# PARAMETROS A DEFINIR: -----------------
###  ENTRADAS
# vector v con la cantidad de neuronas por capa:
vector_num_neuronas = [1,2,1]
cant_capas = len(vector_num_neuronas)
cant_entradas = 2
vector_num_neuronas = np.hstack((cant_entradas, vector_num_neuronas))
#tasa de aprendizaje
v = 0.01
#variable para la tasa
d = 0

#agregar columna de x0 al principo
x0= np.full(cant_filas, -1).reshape(-1, 1)
p0= np.full(cant_filas_p, -1).reshape(-1, 1)
data_TR = np.hstack((x0, data_TR))
data_TE = np.hstack((p0, data_TE))
cant_columnas = cant_columnas+1
cant_salidas = vector_num_neuronas[-1]
#separo las salidas deseadas
yd = data_TR[:,-cant_salidas:]
yd_p = data_TE[:,-cant_salidas:]


#generar pesos aleatorios para comenzar
lista_pesos = np.empty(cant_capas,object)
for i in range(cant_capas):
    lista_pesos[i] = np.random.uniform(-0.5,0.5,(vector_num_neuronas[i+1],vector_num_neuronas[i]+1))
    #agregamos una columna para los pesos w0

## generamos listas
lista_delta = np.empty(cant_capas,object)
lista_salidas = np.empty(cant_capas,object)

# #ITERACIONES ENTRENAMIENTO
it = 0
tasa = 0

#----------------ENTRENAMIENTO-------------------------------------------------------

while  tasa < 0.99 and it <1: #--------- Epocas
    it+=1
    for i in range(cant_filas): #------------- Patron

        #PROPAGACIÓN HACIA ADELANTE
        for j in range(cant_capas): #------------------ Capas
            # Salida LINEAL
            y = 0
            if(j ==0 ):
                y = np.dot(lista_pesos[j],data_TR[i,:cant_columnas-cant_salidas].T) #calculamos y
            else:
                #Se agrega un -1 cuando se propaga hacia delante para utilizarlo como entrada
                y_aux = np.hstack((-1,lista_salidas[j-1]))
                y = np.dot(lista_pesos[j], y_aux.T)
            # Salida NO LINEAL
            y = sigmoide(y,1)
            lista_salidas[j] = y
        
         #PROPAGACIÓN HACIA ATRAS
        for j in range(cant_capas,0,-1):
            derivada_sigmoide = (1+lista_salidas[j-1])*(1-lista_salidas[j-1])
            if(j == cant_capas):
                lista_delta[j-1] = 0.5*(yd[i,:]-lista_salidas[j-1])*derivada_sigmoide
            else:
                #Cortamos la primer columna que contiene los W0
                pesos_aux = lista_pesos[j][:,1:]
                lista_delta[j-1] = 0.5*derivada_sigmoide*np.dot(pesos_aux.T,lista_delta[j])
        
        for j in range(cant_capas):
            delta_W = 0
            if(j==cant_capas-1):
                y_aux = np.hstack((-1,lista_salidas[j-1]))
                delta_W = v*(yd[i,:]-lista_salidas[j])*(1+lista_salidas[j])*(1-lista_salidas[j])*y_aux
            if(j!=0):
                y_aux = np.hstack((-1,lista_salidas[j-1]))
                delta_W = v*np.dot(lista_delta[j+1],lista_pesos[j+1])*(1+lista_salidas[j])*(1-lista_salidas[j])*y_aux
            if(j==0):
                delta_W =v*np.dot(lista_delta[j+1],lista_pesos[j+1])*((1+lista_salidas[j])*(1-lista_salidas[j]))*data_TR[i,:-cant_salidas]
                



print(lista_delta)
