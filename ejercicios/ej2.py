import numpy as np
import matplotlib.pyplot as plt

def graficar(w_r):
    pendiente=-(w_r[1]/w_r[2])
    ordenada = w_r[0]/w_r[2]

    x1 = np.linspace(-1.1, 1.1, 1000, endpoint=False)
    x2 = ordenada + pendiente*x1

    plt.plot(x1, x2, marker='.', linestyle='-', linewidth=0.01)
    plt.ylim(-1.5,1.5)
     
#leer archivo:
#entrenamiento
or_csv = np.loadtxt('C:\\Users\\piliv\\OneDrive\\Documentos\\FACU\\Inteligencia Computacional\\Trabajos prácticos\\Guia1\\XOR_trn.csv', delimiter=',')
[f_or,c_or]= or_csv.shape

#agregar columna al principo
x0_or= np.full(f_or, -1).reshape(-1, 1)
or_csv  = np.hstack((x0_or, or_csv))
c_or+=1

# ENTRENAMIENTO OR -------------------------------
# separo los y deseados
yd_or = or_csv[:,-1]
 
#generar pesos random
w_r = np.zeros(c_or-1) 
for i in range(w_r.size):
    w_r[i] = np.random.uniform(0.0, 1.0)-0.5

#tasa de aprendizaje
v = 0.001
#tasa de desempeño
d1 = 0
#ITERACIONES ENTRENAMIENTO
it = 0

#plot
# Añadir título y etiquetas
plt.title('Gráfico de Líneas')
plt.xlabel('Eje X1')
plt.ylabel('Eje X2')
plt.axhline(0, color='black', linewidth=1)  # Línea horizontal en y=0
plt.axvline(0, color='black', linewidth=1)  # Línea vertical en x=0
plt.grid(True)


#bucle para entrenar
while  d1/f_or< 0.99 and it <50:
    d1=0
    it+=1
    for i in range(f_or):
        y = np.dot(w_r,or_csv[i,:c_or-1]) #calculamos y
        y = np.sign(y)
        for j in range(c_or-1):
            w_r[j] = w_r[j]+ v*(yd_or[i]-y)*or_csv[i,j] #corregimos pesos
        if(i%50 == 1):
            graficar(w_r)

        
    for t in range(f_or):
        y = np.dot(w_r,or_csv[t,:c_or-1]) #calculamos y
        y = np.sign(y)
        if(yd_or[t]== y): d1+=1
plt.scatter(or_csv[:,1], or_csv[:,2], color='g', marker='.')     
plt.show()
