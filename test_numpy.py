import numpy as np
import pandas as pd


from time import time
from sklearn.metrics import confusion_matrix

n_clusters = 2
a = np.random.randint(0, 2, 10000000)
b = np.random.randint(0, 2, 10000000)

t1 = time()

A = confusion_matrix(a, b)

prec = sum([max(A[:, j]) for j in range(0, n_clusters)]) / sum([sum(A[i, :]) for i in range(0, n_clusters)])
rcal = sum([max(A[i, :]) for i in range(0, n_clusters)]) / sum([sum(A[i, :]) for i in range(0, n_clusters)])

t2 = time()
print("Finished in time: %f ." % (t2-t1))
print("prec = %f ; recall = %f ." % (prec, rcal))
