
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Cargar el dataset de dígitos
digits = load_digits()

n_folds= 10
kf = KFold(n_splits=n_folds, shuffle=True)

#Para guardar métricas por clasificador por fold
ACC = [[] for i in range[7]]
