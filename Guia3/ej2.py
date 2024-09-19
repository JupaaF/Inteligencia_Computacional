
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
# from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np


# Cargar el dataset de dígitos
digits = load_digits()

x = digits.data
y = digits.target

n_folds= 10
kf = KFold(n_splits=n_folds, shuffle=True)

#Para guardar métricas por clasificador por fold
ACC = [[] for i in range(7)]

## ------------Particionar datos ---------------

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y, test_size=0.33, random_state=42)

kf_5 = KFold(n_splits=5, shuffle=True )


### kf = 5
for train_index, test_index in kf_5.split(x): 
    
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    ## multicapa
    clf = MLPClassifier(hidden_layer_sizes=(12,6),learning_rate_init=0.005,max_iter=300,activation='logistic',early_stopping=True,validation_fraction=0.3,shuffle=True,random_state=0)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    ACC[0].append(accuracy_score(y_test,y_pred)) ## calcula la tasa de precisión

    ## naive bayes
    clf = GaussianNB()
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    ACC[1].append(accuracy_score(y_test,y_pred)) ## calcula la tasa de precisión

    ## SVM polinomial 
    clf = svm.SVC(kernel='poly',degree=2,coef0=1,gamma='auto')
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    ACC[2].append(accuracy_score(y_test,y_pred)) ## calcula la tasa de precisión

    ## SVM Base Radial
    clf = svm.SVC(kernel='rbf',gamma=1) #---> Gamma: Controla la "influencia" de una sola muestra. Si gamma es muy alto, una muestra individual tendrá un radio de influencia más pequeño, y el modelo podría sobreajustarse. Por el contrario, si es muy bajo, la influencia es mayor y el modelo podría subajustarse.
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    ACC[3].append(accuracy_score(y_test,y_pred)) ## calcula la tasa de precisión

    ## Análisis Discriminante Lineal -->  encuentra un hiperplano que maximiza la distancia entre las medias de las clases mientras minimiza la variación dentro de cada clase.
    clf = LinearDiscriminantAnalysis()
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    ACC[4].append(accuracy_score(y_test,y_pred)) ## calcula la tasa de precisión

    ## Árbol de Decisión
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    ACC[5].append(accuracy_score(y_test,y_pred)) ## calcula la tasa de precisión

    ## KNN
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    ACC[6].append(accuracy_score(y_test,y_pred)) ## calcula la tasa de precisión


media = []
varianza = []

print('------------------ ')
print('Varianzas y medias:')
clasificadores = ['Multicapa','Naive Bayes','SVM polinomial ','SVM Base Radial','Análisis Discriminante Lineal','Árbol de Decisión','KNN']
for i in range (7):
    media.append(np.mean(ACC[i]))
    varianza.append(np.var(ACC[i]))
    print(f'{clasificadores[i]}: media = {media[i]}, varianza = {varianza[i]}')    

#Calculamos el error relativo para comparar el desempeño del mlp respecto al resto:
Er = []
error_rel_mlp = []
for i in range(7):
    Er.append(1 - media[i])

print(Er)

for i in range(1,7):
    error_rel_mlp.append((Er[0]- Er[i])/Er[0])

print('------------------ ')
print('Respecto de MLP:')
for i in range(6):
    print(f'{clasificadores[i+1]} implica una mejora del {np.round(error_rel_mlp[i],2)*100}%')
  


 




