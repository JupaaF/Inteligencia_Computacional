while  tasa < 0.99 and it <50: #--------- Epocas
#     it+=1
#     for i in range(cant_filas): #------------- Patron
#         for j in range(cant_capas): #------------------ Capas
#             # Salida LINEAL
#             if(j ==0 ):
#                 y = np.dot(lista_pesos[j],data_TR[i,:cant_columnas-1].T) #calculamos y
#             else:
#                 y = np.dot(lista_pesos[j],lista_salidas[j-1].T)

#             # Salida NO LINEAL
#             y = sigmoide(y,1)
#             lista_salidas.append(y)

# print(lista_salidas)

