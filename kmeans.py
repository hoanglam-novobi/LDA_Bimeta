import logging
import time

from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans


def evalQuality(y_true, y_pred, n_clusters=2):
    A = confusion_matrix(y_pred, y_true)
    prec = sum([max(A[:, j]) for j in range(0, n_clusters)]) / sum([sum(A[i, :]) for i in range(0, n_clusters)])
    rcal = sum([max(A[i, :]) for i in range(0, n_clusters)]) / sum([sum(A[i, :]) for i in range(0, n_clusters)])
    return prec, rcal


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
    prec = TP / (TP + FP)
    rcal = TP / (TP + FN)
    return prec, rcal


def do_kmeans(top_dist, n_clusters=2, n_workers=2, n_init=100, iters=10000):
    t1 = time.time()
    logging.info("Clustering with Kmeans ...")
    kmeans = KMeans(n_clusters=n_clusters, n_jobs=n_workers, n_init=n_init, max_iter=iters)
    kmeans.fit(top_dist)
    predictions = kmeans.fit_predict(top_dist)
    t2 = time.time()
    logging.info("Finish clustering with Kmeans in %f (s)." % (t2 - t1))
    return predictions
