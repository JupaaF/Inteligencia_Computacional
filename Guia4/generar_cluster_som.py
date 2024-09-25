import numpy as np
def gen_cluster_iris(pesos,x):
    i_ganadora = 0
    j_ganadora = 0
    [cant_filas,cant_columnas]= x.shape
    [N1,N2] = pesos.shape
    categoria = x[:,-3:]
    for j in range(len(categoria)):
        if (categoria[j] == [1,-1,-1]).all():
            categoria[j] = [0, 0, 0]
        if (categoria[j] == [-1,1,-1]).all():
            categoria[j] = [1, 0, 0]
        if (categoria[j] == [-1,-1,1]).all():
            categoria[j] = [2, 0, 0]
    clusters = np.empty((N1,N2),object)
    for i in range(N1):
        for j in range(N2):
            clusters[i,j] = [0,0,0]
    for p in range(cant_filas): # Para cada patron x: --------
        menor_d = 1000000
        #Recorro pesos para encontrar neurona ganadora
        for i in range(N1):
            for j in range(N2):
                #Calculo de distancia
                distancia = np.sum((x[p,:-3] - pesos[i,j]) ** 2) 
                #Neurona ganadora
                if(distancia<=menor_d):
                    menor_d = distancia
                    i_ganadora=i
                    j_ganadora=j
        clusters[i_ganadora,j_ganadora][int(categoria[p,0])] +=1
    clasificador_final = np.empty((N1,N2),int)
    for i in range(N1):
        for j in range(N2):
            if (clusters[i,j]==[0,0,0]):
                clasificador_final[i,j] = 3
                continue
            clasificador_final[i,j] = clusters[i,j].index(max(clusters[i,j]))
    v_medias = np.empty(3,object)
    for i in range(3):
        suma_v = [0,0,0,0]
        cant_v = 0
        for j in range(N1):
            for k in range(N2):
                if(clasificador_final[j,k]==i):
                    suma_v+=pesos[j,k]
                    cant_v+=1
        v_medias[i] = suma_v/cant_v
    return(v_medias)
