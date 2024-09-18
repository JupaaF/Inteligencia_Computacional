
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_wine
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
# from tabulate import tabulate
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

ACC = [[] for i in range(3)]
y_pred = np.empty(3,object)

for train_index, test_index in kf.split(x): 
    
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    ## multicapa
    clf = MLPClassifier(hidden_layer_sizes=(12,6),learning_rate_init=0.005,max_iter=300,activation='logistic',early_stopping=True,validation_fraction=0.3,shuffle=True,random_state=0)
    clf.fit(x_train,y_train)
    y_pred[0] = clf.predict(x_test)
    ACC[0].append(accuracy_score(y_test,y_pred[0])) ## calcula la tasa de precisión

    ## naive bayes
    clf = GaussianNB()
    clf.fit(x_train,y_train)
    y_pred[1] = clf.predict(x_test)
    ACC[1].append(accuracy_score(y_test,y_pred[1])) ## calcula la tasa de precisión

    ## SVM polinomial 
    clf = svm.SVC(kernel='poly',degree=2,coef0=1,gamma='auto')
    clf.fit(x_train,y_train)
    y_pred[2] = clf.predict(x_test)
    ACC[2].append(accuracy_score(y_test,y_pred[2])) ## calcula la tasa de precisión

print(y_pred)
yens = np.zeros(len(y_pred[0]))

for j in range (len(y_pred[0])):
    for i in range (3):
        yens[j] += y_pred[i][j]
    yens[j] = round(yens[j]/3)


print(f'promedio salidas {yens}')


# # Mostrar información del dataset
# print("Características del dataset:", feature_names)
# print("Nombres de las clases:", target_names)
# print("Forma de las características:", x.shape)
# print("Forma de las etiquetas:", y.shape)

# # Mostrar las primeras filas de los datos
# print("\nPrimeras filas de las características:")
# print(x[:5])

# print("\nPrimeras etiquetas:")
# print(y[:5])

