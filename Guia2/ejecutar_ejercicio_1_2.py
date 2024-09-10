cant_it = 3000
nom_training = 'Guia2/concent_trn.csv'
nom_testing = 'Guia2/concent_tst.csv'
vec_neuronas = [6,1]
cant_entradas = 2
tasa_aprendizaje = 0.01
tasa_corte = 0.99

from funcion_multicapa import multicapa

multicapa(cant_it,nom_training,nom_testing,vec_neuronas,cant_entradas,tasa_aprendizaje,tasa_corte)

