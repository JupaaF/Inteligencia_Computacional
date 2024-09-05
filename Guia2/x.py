import numpy as np
import matplotlib.pyplot as plt
import math 

def winner_takes_all(v):
    i = np.argmax(v)
    for j in range(len(v)):
        if(j==i):
            v[j]=1
        else:
            v[j]=-1
    return v


v = [[0.7,0.1,-0.2],[0.7,0.1,-0.2],[0.7,0.1,-0.2]]

v = winner_takes_all(v)
print(v)