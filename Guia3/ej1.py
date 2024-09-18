
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Cargar el dataset de dígitos
digits = load_digits()

#Particionar datos
x_trn, x_tst, y_trn, y_tst = train_test_split(digits.data, digits.target, test_size=0.33, random_state=42) #???

#Definir modelo
clf = MLPClassifier(hidden_layer_sizes=(20,10),learning_rate_init=0.005,max_iter=300,activation='logistic',early_stopping=True,validation_fraction=0.3,shuffle=True,random_state=0)

#Entrenar modelo
clf.fit(x_trn,y_trn)
#Evaluar modelo
y_pred = clf.predict(x_tst)

#Cálculo de métricas
ACC = accuracy_score(y_tst,y_pred)
MC = confusion_matrix(y_tst,y_pred,labels=digits.target_names)

disp = ConfusionMatrixDisplay(confusion_matrix=MC, display_labels=digits.target_names)
fig, ax = plt.subplots(figsize=(8,8))
disp.plot(ax=ax, cmap="Blues", values_format='', colorbar=None)
print('ACC',ACC)
