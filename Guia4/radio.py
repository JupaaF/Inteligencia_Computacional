
def radio(r, pesos, fila, columna,delta_w,tamanio_fila,tamanio_columna, id ):
    
    if (fila >= tamanio_fila or (fila== -1)):
        return
    
    for i in range (-r,r+1):
        if((columna+i < tamanio_columna) and (columna+i != -1)):
            pesos[fila,columna+i] += delta_w
    
    if(r == 0): 
        return   
        
    if(id == 0):
        radio(r-1,pesos,fila+1,columna,delta_w,tamanio_fila,tamanio_columna,1)
        radio(r-1,pesos,fila-1,columna,delta_w,tamanio_fila,tamanio_columna,-1)
    if(id == 1):
        radio(r-1,pesos,fila+1,columna,delta_w,tamanio_fila,tamanio_columna,1)
    if(id == -1):
        radio(r-1,pesos,fila-1,columna,delta_w,tamanio_fila,tamanio_columna,-1)

pesos = np.ones()
