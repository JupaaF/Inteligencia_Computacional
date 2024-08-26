import numpy as np
import matplotlib.pyplot as plt

#leer archivo:
#entrenamiento
or_csv = np.loadtxt('C:\\Users\\piliv\\OneDrive\\Documentos\\FACU\\Inteligencia Computacional\\Trabajos prácticos\\Guia1\\OR_trn.csv', delimiter=',')
#prueba
xor_csv = np.loadtxt('C:\\Users\\piliv\\OneDrive\\Documentos\\FACU\\Inteligencia Computacional\\Trabajos prácticos\\Guia1\\XOR_trn.csv', delimiter=',')
[f_or,c_or]= or_csv.shape
[f_xor,c_xor]= xor_csv.shape

#agregar columna al principo
x0_or= np.full(f_or, -1).reshape(-1, 1)
x0_xor= np.full(f_xor, -1).reshape(-1, 1)
or_csv  = np.hstack((x0_or, or_csv))
xor_csv  = np.hstack((x0_xor, xor_csv))
c_or+=1
c_xor+=1

# ENTRENAMIENTO OR -------------------------------
#separo los y deseados
# yd_or = or_csv[:,-1]
 
# #generar pesos random
# w_r = np.zeros(c_or-1) 
# for i in range(w_r.size):
#     w_r[i] = np.random.uniform(0.0, 1.0)-0.5

# #tasa de aprendizaje
# v = 0.01
# #tasa de desempeño
# d1 = 0
# #ITERACIONES ENTRENAMIENTO
# it = 0

# #bucle para entrenar
# while  d1/f_or< 0.99 and it <500:
#     d1=0
#     it+=1
#     for i in range(f_or):
#         y = np.dot(w_r,or_csv[i,:c_or-1]) #calculamos y
#         for j in range(c_or-1):
#             w_r[j] = w_r[j]+ v*(yd_or[i]-y)*or_csv[i,j] #corregimos pesos
  
#     for t in range(f_or):
#         y = np.dot(w_r,or_csv[t,:c_or-1]) #calculamos y
#         if(yd_or[t]== np.sign(y)): d1+=1


# pendiente=-(w_r[1]/w_r[2])

# ordenada = w_r[0]/w_r[2]

# x1 = np.linspace(-1.1, 1.1, 1000, endpoint=False)
# x2 = ordenada + pendiente*x1

# plt.plot(x1, x2, marker='o', linestyle='-', color='b')
# plt.scatter(or_csv[:,1], or_csv[:,2], color='g', marker='.')

# # Añadir título y etiquetas
# plt.title('Gráfico de Líneas')
# plt.xlabel('Eje X1')
# plt.ylabel('Eje X2')
# plt.axhline(0, color='black', linewidth=1)  # Línea horizontal en y=0
# plt.axvline(0, color='black', linewidth=1)  # Línea vertical en x=0
# plt.grid(True)
# # Mostrar el gráfico
# plt.show()

# ENTRENAMIENTO xOR -------------------------------

#separo los y deseados
yd_xor = xor_csv[:,-1]
 
#generar pesos random
w_r = np.zeros(c_xor-1) 
for i in range(w_r.size):
    w_r[i] = np.random.uniform(0.0, 1.0)-0.5

#tasa de aprendizaje
v = 0.01
#tasa de desempeño
d1 = 0
#ITERACIONES ENTRENAMIENTO
it = 0

#bucle para entrenar
while  d1/f_xor < 0.99 and it <500:
    d1=0
    it+=1
    for i in range(f_xor):
        y = np.dot(w_r,xor_csv[i,:c_xor-1]) #calculamos y
        for j in range(c_xor-1):
            w_r[j] = w_r[j]+ v*(yd_xor[i]-y)*xor_csv[i,j] #corregimos pesos
  
    for t in range(f_xor):
        y = np.dot(w_r,xor_csv[t,:c_xor-1]) #calculamos y
        if(yd_xor[t]== np.sign(y)): d1+=1


pendiente=-(w_r[1]/w_r[2])

ordenada = w_r[0]/w_r[2]

x1 = np.linspace(-1.1, 1.1, 1000, endpoint=False)
x2 = ordenada + pendiente*x1

plt.plot(x1, x2, marker='o', linestyle='-', color='b')
plt.scatter(xor_csv[:,1], xor_csv[:,2], color='g', marker='.')

# Añadir título y etiquetas
plt.title('Gráfico de Líneas')
plt.xlabel('Eje X1')
plt.ylabel('Eje X2')
plt.axhline(0, color='black', linewidth=1)  # Línea horizontal en y=0
plt.axvline(0, color='black', linewidth=1)  # Línea vertical en x=0
plt.grid(True)
plt.show()