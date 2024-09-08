cant_it = 50000
nom_training = 'Guia2/concent_trn.csv'
nom_testing = 'Guia2/concent_tst.csv'
vec_neuronas = [10,4,1]
cant_entradas = 2
tasa_aprendizaje = 0.001
tasa_corte = 0.95

from funcion_multicapa import multicapa

multicapa(cant_it,nom_training,nom_testing,vec_neuronas,cant_entradas,tasa_aprendizaje,tasa_corte)

