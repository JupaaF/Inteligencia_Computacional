
import numpy as np

def radio(r, pesos, fila, columna,delta_w,tamanio_fila,tamanio_columna, id ):
    
    if (fila >= tamanio_fila or (fila== -1)):
        return
    
    for i in range (-r,r+1):
        if((columna+i < tamanio_columna) and (columna+i != -1)):
            pesos[fila,columna+i] += delta_w
            print(pesos[fila,columna+i])
    
    if(r == 0): 
        return   
        
    if(id == 0):
        radio(r-1,pesos,fila+1,columna,delta_w,tamanio_fila,tamanio_columna,1)
        radio(r-1,pesos,fila-1,columna,delta_w,tamanio_fila,tamanio_columna,-1)
    if(id == 1):
        radio(r-1,pesos,fila+1,columna,delta_w,tamanio_fila,tamanio_columna,1)
    if(id == -1):
        radio(r-1,pesos,fila-1,columna,delta_w,tamanio_fila,tamanio_columna,-1)



### prueba
# N1 = 4
# N2 = 4


# delta_w = np.array([1,1])

# pesos = np.empty((N1,N2), object)

# for i in range(N1):
#     for j in range(N2):
#         # pesos[i,j] = np.random.uniform(-0.5,0.5,(2,))
#         pesos[i,j] = [1.0,1.0]


# for i in range (N1):
#     for j in range (N2):
#         print(f'Fila: {i} Columna: {j} Peso: {pesos[i,j]}')
    
# radio(2, pesos,3,2,delta_w,N1,N2,0)
# print('ss')
# for i in range (N1):
#     for j in range (N2):
#         print(f'Fila: {i} Columna: {j} Peso: {pesos[i,j]}')


# radio(3, pesos,2,3,delta_w,N1,N2,0)
# print('ss')
# for i in range (N1):
#     for j in range (N2):
#         print(f'Fila: {i} Columna: {j} Peso: {pesos[i,j]}')

