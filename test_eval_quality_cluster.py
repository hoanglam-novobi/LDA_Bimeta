import time
import multiprocessing as mp
import numpy as np

from multiprocessing import Pool, Value, Array
from kmeans import evalQualityCluster, parallel_evalQualityCluster
from read_fasta import create_labels


def eval_quality(y_true, y_pred, low_idx, high_idx):
    selected_ytrue = y_true[low_idx:high_idx]
    selected_ypred = y_pred[low_idx:high_idx]


if __name__ == "__main__":
    y_true, n_cluster = create_labels('R1')
    print("Number of cluster: %d" % n_cluster)
    size = len(y_true)
    y_pred = np.random.randint(0, 2, size)

    t1 = time.time()
    n_workers = 40

    prec, recall = parallel_evalQualityCluster(y_true, y_pred, n_workers=n_workers, n_clusters=n_cluster)

    t2 = time.time()
    print("Finised time in %f (s)" % (t2 - t1))
