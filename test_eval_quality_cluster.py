import time
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

actual_labels = {
    'R1': [42189, 40771],
    'R2': [38664, 38629],
    'R3': [47457, 45810],
    'R4': [16447, 18010],
    'R5': [22027, 18016],
    'R6': [27029, 43521],
    'R7': [19473, 19291, 251709],
    'S1': [44405, 51962],
    'S2': [114177, 81162],
    'S3': [89724, 249001],
    'S4': [57260, 318042],
    'S5': [114250, 81063, 130087],
    'S6': [172675, 317955, 222758],
    'S7': [141928, 75183, 126183, 588088, 722168]
}


def evalQuality(y_true, y_pred, n_clusters=2):
    A = confusion_matrix(y_pred, y_true)
    prec = sum([max(A[:, j]) for j in range(0, n_clusters)]) / sum([sum(A[i, :]) for i in range(0, n_clusters)])
    rcal = sum([max(A[i, :]) for i in range(0, n_clusters)]) / sum([sum(A[i, :]) for i in range(0, n_clusters)])
    return prec, rcal


def create_labels(name):
    n_cluster = len(actual_labels[name])
    labels = []
    for idx, value in enumerate(actual_labels[name]):
        labels += [idx] * value
    return labels, n_cluster


# load dataset
DATA_DIR = 'D:/HOCTAP/HK182/Luanvantotnghiep/sourcecode/dataset/'
lda_prediction_data = pd.read_csv(DATA_DIR + 'R4lda_prediction_result.csv')
lda_bimeta_prediction_data = pd.read_csv(DATA_DIR + 'R4lda_bimeta_prediction_result.csv')

# convert to df
lda_bimeta_prediction_df = pd.DataFrame(lda_bimeta_prediction_data)
lda_prediction_df = pd.DataFrame(lda_prediction_data)

actual = lda_prediction_df['actual'].values
lda_predictions = lda_prediction_df['prediction'].values
lda_bimeta_predictions = lda_bimeta_prediction_df['prediction'].values

lda_prec, lda_recall = evalQuality(actual, lda_predictions)
print("LDA: Prec = %f ; Recall = %f ." % (lda_prec, lda_recall))

lda_bimeta_prec, lda_bimeta_recall = evalQuality(actual, lda_bimeta_predictions)
print("LDA + Bimeta: Prec = %f ; Recall = %f ." % (lda_bimeta_prec, lda_bimeta_recall))
