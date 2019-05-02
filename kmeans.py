import logging
import time
import numpy as np
import multiprocessing as mp

from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from bimeta import assign_cluster

# CONFIG FOR LOGGING MEMORY_PROFILER
import sys
from memory_profiler import profile, LogFile

sys.stdout = LogFile(__name__)


@profile
def evalQuality(y_true, y_pred, n_clusters=2):
    A = confusion_matrix(y_pred, y_true)
    prec = sum([max(A[:, j]) for j in range(0, n_clusters)]) / sum([sum(A[i, :]) for i in range(0, n_clusters)])
    rcal = sum([max(A[i, :]) for i in range(0, n_clusters)]) / sum([sum(A[i, :]) for i in range(0, n_clusters)])
    return prec, rcal


@profile
def evalQualityCluster(y_true, y_pred, n_clusters=2):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(0, len(y_true) - 1):
        for j in range(i + 1, len(y_true)):
            if y_true[i] == y_true[j] and y_pred[i] == y_pred[j]:
                TP += 1
            elif y_true[i] == y_true[j] and y_pred[i] != y_pred[j]:
                FN += 1
            elif y_true[i] != y_true[j] and y_pred[i] == y_pred[j]:
                FP += 1
            else:
                TN += 1
    return TP, FP, TN, FN


@profile
def parallel_evalQualityCluster(y_true, y_pred, n_workers=3, n_clusters=2):
    TP, FP, TN, FN = 0, 0, 0, 0
    pool = mp.Pool(n_workers)

    for input1, input2 in zip(np.array_split(y_true, n_workers), np.array_split(y_pred, n_workers)):
        _res = pool.apply_async(evalQualityCluster, args=(input1, input2, n_clusters))
        result = _res.get()
        TP += result[0]
        FP += result[1]
        TN += result[2]
        FN += result[3]

    prec = TP / (TP + FP)
    rcal = TP / (TP + FN)
    return prec, rcal


@profile
def do_kmeans(top_dist, seeds=np.array([]), seeds_dict=None, n_clusters=2, n_workers=2, n_init=100, iters=10000):
    t1 = time.time()
    logging.info("Clustering with Kmeans ...")
    kmeans = KMeans(n_clusters=n_clusters, n_jobs=n_workers, n_init=n_init, max_iter=iters)

    if seeds.size == 0:
        predictions = list(kmeans.fit_predict(top_dist))
    else:
        seed_clusters = list(kmeans.fit_predict(seeds))
        predictions = assign_cluster(mean_vector_clusters=seed_clusters, seeds=seeds_dict)

    t2 = time.time()
    logging.info("Finish clustering with Kmeans in %f (s)." % (t2 - t1))
    return predictions
