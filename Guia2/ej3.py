import numpy as np
import matplotlib.pyplot as plt
import math 

def winner_takes_all(v):
    i = np.argmax(v)
    for j in range(len(v)):
        if(j==i):
            v[j]=1
        else:
            v[j]=-1
    return v

def sigmoide(z,a):
    for i in range(len(z)):
        z[i] = (1-math.e**(-a*z[i]))/(1+math.e**(-a*z[i]))
    return z

#leer archivos: ----------------
#entrenamiento
data_TR = np.loadtxt('Guia2/irisbin_trn.csv', delimiter=',')
#prueba
data_TE = np.loadtxt('Guia2/irisbin_tst.csv', delimiter=',')
[cant_filas,cant_columnas]= data_TR.shape
[cant_filas_p,cant_columnas_p]= data_TE.shape



# PARAMETROS A DEFINIR: -----------------
###  ENTRADAS
# vector v con la cantidad de neuronas por capa:
vector_num_neuronas = [4,3]
cant_capas = len(vector_num_neuronas)
cant_entradas = 4
vector_num_neuronas = np.hstack((cant_entradas, vector_num_neuronas))
#tasa de aprendizaje
v = 0.1
#variable para la tasa
d = 0
cant_iteraciones = 1000

#agregar columna de x0 al principo
x0= np.full(cant_filas, -1).reshape(-1, 1)
p0= np.full(cant_filas_p, -1).reshape(-1, 1)
data_TR = np.hstack((x0, data_TR))
data_TE = np.hstack((p0, data_TE))
cant_columnas = cant_columnas+1
cant_salidas = vector_num_neuronas[-1]
#separo las salidas deseadas
yd = data_TR[:,-cant_salidas:]
yd_TE = data_TE[:,-cant_salidas:]


#generar pesos aleatorios para comenzar
lista_pesos = np.empty(cant_capas,object)
for i in range(cant_capas):
    lista_pesos[i] = np.random.uniform(-0.5,0.5,(vector_num_neuronas[i+1],vector_num_neuronas[i]+1))
    #agregamos una columna para los pesos w0

## generamos listas
lista_delta = np.empty(cant_capas,object)
lista_salidas = np.empty(cant_capas,object)
lista_salidas_TE = np.empty(cant_capas,object)

# #ITERACIONES ENTRENAMIENTO
it = 0
tasa = 0
d_TR = 0
d_TE = 0
mse = np.empty(cant_iteraciones,object)
#----------------ENTRENAMIENTO-------------------------------------------------------

while  tasa < 0.95 and it <cant_iteraciones: #--------- Epocas

    for i in range(cant_filas): #------------- Patron
        d_TR = 0
        #PROPAGACIÓN HACIA ADELANTE
        for j in range(cant_capas): #------------------ Capas
            # Salida LINEAL
            y = []
            if(j ==0 ):
                # y = np.dot(lista_pesos[j],data_TR[i,:cant_columnas-cant_salidas].T) #calculamos y
                y = lista_pesos[j]@data_TR[i,:cant_columnas-cant_salidas].T
            else:
                #Se agrega un -1 cuando se propaga hacia delante para utilizarlo como entrada x0
                y_aux = np.hstack((-1,lista_salidas[j-1]))
                # y = np.dot(lista_pesos[j], y_aux.T)
                y = lista_pesos[j]@y_aux.T
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
                lista_delta[j-1] = 0.5*derivada_sigmoide*(pesos_aux.T@lista_delta[j])
        
        #ACTUALIZACIÓN DE LOS PESOS
        for j in range(cant_capas):
            #Matriz delta_W --> de (cantidad de neuronas*cantidad de entradas)
            delta_W = np.empty((vector_num_neuronas[j+1],vector_num_neuronas[j]+1),object)
          
            if(j==0):
                #Calculamos el delta con las entradas x
                entradas_aux = data_TR[i,:-cant_salidas]
                delta_W = np.outer((v*lista_delta[j]).T,entradas_aux)
                
            else:
                #Calculamos el delta con las salidas de la capa anterior
                y_aux = np.hstack((-1,lista_salidas[j-1]))
                delta_W = np.outer((v*lista_delta[j]).T,y_aux)

            lista_pesos[j] = lista_pesos[j] + delta_W
        
#-----------------------------------------------------------------------
    
    #Calculamos de nuevo la salida para obtener el desempeño
    salidas_trn = np.empty([cant_filas,cant_salidas],object)

    for i in range(cant_filas):
        for j in range(cant_capas): #------------------ Capas
            # Salida LINEAL
            y = []
            if(j ==0 ):
                # y = np.dot(lista_pesos[j],data_TR[i,:cant_columnas-cant_salidas].T) #calculamos y
                y = lista_pesos[j]@data_TR[i,:cant_columnas-cant_salidas].T
            else:
                #Se agrega un -1 cuando se propaga hacia delante para utilizarlo como entrada x0
                y_aux = np.hstack((-1,lista_salidas[j-1]))
                # y = np.dot(lista_pesos[j], y_aux.T)
                y = lista_pesos[j]@y_aux.T
            # Salida NO LINEAL
            y = np.sign(y) #------------> usamos la signo para que concuerde con las y deseadas
            lista_salidas[j] = y
            #hay que comparar con la salida deseada

        w_t_a = winner_takes_all(lista_salidas[-1])
        if(np.array_equal(yd[i],w_t_a)):
            d_TR+=1
        salidas_trn[i]=w_t_a
    
    e = yd-salidas_trn
    mse[it] = e@e.T/cant_filas

    tasa= d_TR/cant_filas




#PRUEBA -------------------------------------------------------------------------
salidas_prueba = np.empty(cant_filas_p,object)

for i in range(cant_filas_p): #-------------------- Patrones
    
    for j in range(cant_capas): #-------------------- Capas
        # Salida LINEAL
        y = 0
        if(j ==0 ):
            # y = np.dot(lista_pesos[j],data_TE[i,:-cant_salidas].T) #calculamos y
            y = lista_pesos[j]@data_TE[i,:-cant_salidas].T
        else:
            #Se agrega un -1 cuando se propaga hacia delante para utilizarlo como entrada x0
            y_aux = np.hstack((-1,lista_salidas[j-1]))
            # y = np.dot(lista_pesos[j], y_aux.T)
            y = lista_pesos[j]@y_aux.T
        # Salida NO LINEAL

        y = np.sign(y) #------------> usamos la signo para que concuerde con las y deseadas
        lista_salidas[j] = y
        #hay que comparar con la salida deseada
    
    if(np.array_equal(yd_TE[i],lista_salidas[-1])):
        d_TE+=1

    salidas_prueba[i] = lista_salidas[-1]
    
    it+=1


tasa_TE= d_TE/cant_filas_p



print(f'Desempeño Entrenamiento: {100*tasa} ')
print(f'Iteraciones Entrenamiento: {it}')
print(f'Desempeño Prueba: {100*tasa_TE} ')