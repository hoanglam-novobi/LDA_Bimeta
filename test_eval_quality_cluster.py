import time
import multiprocessing as mp
import numpy as np

from multiprocessing import Pool, Value, Array
from kmeans import evalQualityCluster
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
    pool = Pool(n_workers)
    for input1, input2 in zip(np.array_split(y_true, n_workers), np.array_split(y_pred, n_workers)):
        print("y_true: ", input1)
        print("y_pred: ", input2)
        res = pool.apply_async(evalQualityCluster, args=(input1, input2, n_workers))
        print(res.get())

    t2 = time.time()
    print("Finised time in %f (s)" % (t2 - t1))
