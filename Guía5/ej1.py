import numpy as np
import torch
from copy import deepcopy
from torch import nn
from torch import optim
from torch import FloatTensor
from torch.utils.data import Dataset, DataLoader

# Leer archivos: -------------------
data_trn = np.loadtxt('Guía5/XOR_trn.csv', delimiter=',')
data_tst = np.loadtxt('Guía5/XOR_tst.csv', delimiter=',')

# Datos como tensores de pytorch (matrices como las de np con otras propiedades),
#   tiene espacio para guardar las derivadas (el gradiente)
X_train_tensor = FloatTensor(data_trn[:,:-1])
y_train_tensor = FloatTensor(data_trn[:,-1])

X_test_tensor = FloatTensor(data_tst[:,:-1])
y_test = data_tst[:,-1]

#definicion del modelo para resolver XOR
class MLP(nn.Module):

    #==============================
    def __init__(self): #--> donde se construye el modelo(objeto)
        super(MLP, self).__init__()

        self.capa1 = nn.Linear(2,2) #2 entradas, 2 salidas
        # --> el bias se maneja por separado (un vector que se suma a la salida)
        # (por defecto está activado, si no lo queres, es un boolean q se le pasa a Linear)
        self.capa2 = nn.Linear(2,2) #2 entradas, 1 salida

        self.activacion = nn.Tanh()

    #==============================
    def forward(self,x): #--> define cómo cuando llega un patron, se va a procesar la info y calcular la salida
        y= self.capa1(x)
        y= self.activacion(y)

        y= self.capa2(y)
        y= self.activacion(y)

        return y
    
#entrenamiento
mlp_pyt = MLP() #Instanciamos el modelo
loss_criterion = nn.MSELoss() #criterio para medir el error
optimizer = optim.SGD(mlp_pyt.parameters(),lr=0.1,momentum=0.9)# metodo de gradiente y velocidad de aprendizaje
                    #parametros q quiero que actualize, vel, momento(suaviza o acelera las modif de los pesos)

mlp_pyt.train()
#PASADA BATCH - una misma actualizacion para todos los pesos?
for epoch in range(1024):
    optimizer.zero_grad()                                   #inicializacion de los gradientes (inicializar el lugar para guardar los gradientes)
    y_pred = mlp_pyt(X_train_tensor)                        #pasada hacia adelante
    loss = loss_criterion(y_pred.squeeze(),y_train_tensor)  #se miden los errores
    loss.backward()                                         #pasada hacia atras
    optimizer.step()                                        #ajuste de pesos

#prueba
mlp_pyt.eval()
y_test_pred = mlp_pyt(X_test_tensor)               #pasada hacia adelante
y_test_pred = y_test_pred.detach().round()         #redondea a +1/-1

acc_pyt = sum(y_test_pred.squeeze().numpy() == y_test())/len(y_test) #exactitud
print(acc_pyt)
    





