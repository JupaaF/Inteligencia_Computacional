
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
# Cargar el dataset de dígitos
digits = load_digits()

x = digits.data
y = digits.target

## ------------Particionar datos ---------------
## 1 partición
x_trn, x_tst, y_trn, y_tst = train_test_split(x, y, test_size=0.33, random_state=42)
#test_size= 
# random_state = valor para inicializar el generador aleatorio
# si se usa el mismo numero garantiza reproducir la división
## 5 particiones
kf_5 = KFold(n_splits=5, shuffle=True )
ACC_5 = []


## 10 particiones
kf_10 = KFold(n_splits=10, shuffle=True )
ACC_10 = []


##--------#Definir modelo ----------------
clf = MLPClassifier(hidden_layer_sizes=(12,6),learning_rate_init=0.005,max_iter=300,activation='logistic',early_stopping=True,validation_fraction=0.3,shuffle=True,random_state=0)
## hidden_layer_sizes() : define estructura capas ocultas
## activation : función de activación
## learning_rate_init : tasa de aprendizaje
## max_iter : cantidad máxima de épocas
## early_stoping_true: detiene si el rendimiento sobre los datos de validación deja de mejorar 
##                     durante un número determinado de iteraciones 
## validation_fraction: define la cantidad de datos utilizados para validar
## shuffle: para que sean mezclados
## random_state : inicializa los datos con el mismo valor para que siempre sea el mismo resultado


#Entrenar modelo
clf.fit(x_trn,y_trn)
#Evaluar modelo
y_pred = clf.predict(x_tst)
## devuelve arreglo de etiquetas predichas

#Cálculo de métricas
ACC = accuracy_score(y_tst,y_pred) ## calcula la tasa de precisión
MC = confusion_matrix(y_tst,y_pred,labels=digits.target_names)

## crea una clase de matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=MC, display_labels=digits.target_names)

fig, ax = plt.subplots(figsize=(8,8))

## ejecuta función para mostrar
disp.plot(ax=ax, cmap="Blues", values_format='', colorbar=None)
print('ACC',ACC)
plt.show()



### kf = 5
for train_index, test_index in kf_5.split(x): 
    
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    clf = MLPClassifier(hidden_layer_sizes=(12,6),learning_rate_init=0.005,max_iter=300,activation='logistic',early_stopping=True,validation_fraction=0.3,shuffle=True,random_state=0)

    #Entrenar modelo
    clf.fit(x_train,y_train)
    #Evaluar modelo
    y_pred = clf.predict(x_test)
    ## devuelve arreglo de etiquetas predichas

    #Cálculo de métricas
    ACC_5.append(accuracy_score(y_test,y_pred)) ## calcula la tasa de precisión
 
media_5 = np.mean(ACC_5)
varianza_5 = np.var(ACC_5)

### kf = 10
for train_index, test_index in kf_10.split(x): 

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    clf = MLPClassifier(hidden_layer_sizes=(12,6),learning_rate_init=0.005,max_iter=300,activation='logistic',early_stopping=True,validation_fraction=0.3,shuffle=True,random_state=0)

    #Entrenar modelo
    clf.fit(x_train,y_train)
    #Evaluar modelo
    y_pred = clf.predict(x_test)
    ## devuelve arreglo de etiquetas predichas

    #Cálculo de métricas
    ACC_10.append(accuracy_score(y_test,y_pred)) ## calcula la tasa de precisión

media_10 = np.mean(ACC_10)
varianza_10 = np.var(ACC_10)

print(f'5: media {media_5}, varianza: {varianza_5}')
print(f'10: media {media_10}, varianza: {varianza_10}')

