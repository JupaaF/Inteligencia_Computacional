#Inicializar población --> 1) crear individuos como cadenas de bits

#Evaluar-->1) traducir cadena de bit a parametros
#       -->2) hacer funcion de fitness para evaluar
#       -->3) guardar valores fitness

#Repetir hasta cumplir aptitud
    #Generar nueva poblacion
        #seleccionar padres 
            #a) ruelta:  f = [4.5 2 3 0.5] divido sobre la suma
            #          =[0.45 0.2 0.3 0.05]
            # => hago sumo hasta cada uno: [0.45 0.6 0.95 1]
            #      tiro un nro random entre 0 y 1 y si es 0<x<0.45, elijo el primero, etc etc
            #b) ventana:  ordeno por fitness, elijo ventana, saco uno aleat de ese rango
            #              y repito para otra ventana pequeña (puede salir 2 o más veces el mismo individio)   
            #c) competencia:
        #cruzas
        #reemplazamos toda la poblacion
        #mutacion a todos los indiv (probabilidad muy baja)
    #evaluar fitness y guardar valores



