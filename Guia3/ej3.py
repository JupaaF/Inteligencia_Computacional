
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_wine
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
# from tabulate import tabulate
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
import numpy as np


# Cargar el dataset de dígitos
wine = load_wine()

x = wine.data
y = wine.target

feature_names = wine.feature_names  # Nombres de las características
target_names = wine.target_names  # Nombres de las etiquetas

n_folds= 5
kf = KFold(n_splits=n_folds, shuffle=True)

ACC = [[] for i in range(2)]
y_pred = np.empty(2,object)

#### parametros a cambiar
clasificador_base = DecisionTreeClassifier()
est = 50

for train_index, test_index in kf.split(x): 
    
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    ## Baggin
    bagg = BaggingClassifier(clasificador_base,n_estimators=est,max_samples=0.5,max_features=0.5, n_jobs=5)
    ## n_estimators : n° de clasificadores base que se entrenarán
    ## max_samples : define proporción de muestras del conjunto de datos origan
    ##              osea en este caso entrena utilizando el 50% de los datos de entrenamiento
    ## n_jobs: especifica el n° de nucleos de CPU que se utilizarán para entrenar los clasificadores en paralelo
    bagg.fit(x_train,y_train)
    y_pred[0] = bagg.predict(x_test)
    ACC[0].append(accuracy_score(y_test,y_pred[0])) ## calcula la tasa de precisión

    ## Ada Boost
    ada = AdaBoostClassifier(n_estimators=est)
    ada.fit(x_train,y_train)
    y_pred[1] = ada.predict(x_test)
    ACC[1].append(accuracy_score(y_test,y_pred[1])) ## calcula la tasa de precisión

promedio_tasa = np.mean(ACC,axis=1)
std_tasa = np.var(ACC,axis=1)

print(promedio_tasa)
print(f'tasa y varianza baggin : {promedio_tasa[0]}, {std_tasa[0]}')
print(f'tasa y varianza adaboost : {promedio_tasa[1]}, {std_tasa[1]}')
