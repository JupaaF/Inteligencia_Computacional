import numpy as np

def funcion_modif_segun_entorno(r, pesos, modif_pesos, i_ganadora,j_ganadora,N1,N2,v,xx):
        #Recorrer pesos para modificar los que esten dentro del entorno de la i,j ganadora
        for i in range(N1):
            for j in range(N2):
                # Calcular la distancia de Manhattan desde (i_ganadora, j_ganadora) a (i, j)
                distancia = abs(i - i_ganadora) + abs(j - j_ganadora) 
                
                # Si la distancia es menor o igual a R, modificar peso
                if(distancia <= r):
                    pesos[i, j] = pesos[i, j] + v*(xx-pesos[i,j])