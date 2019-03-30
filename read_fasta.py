import itertools as it
import logging
import time

from Bio import SeqIO

# CONFIG FOR LOGGING MEMORY_PROFILER
import sys
from memory_profiler import profile, LogFile

sys.stdout = LogFile(__name__)

# LABELS for datasets
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


@profile
def read_fasta_file(file_name, pair_end=True):
    t1 = time.time()
    # read sequence from fasta file
    all_reads = list(map(lambda item: str(item.seq), list(SeqIO.parse(file_name, 'fasta'))))
    logging.info("Reading %d reads from the file %s." % (len(all_reads), file_name))
    if pair_end:
        reads = []
        for i in range(0, len(all_reads), 2):
            reads.append(all_reads[i] + all_reads[i + 1])
        all_reads = reads
    t2 = time.time()
    logging.info("Finished reading data in %f (s)" % (t2 - t1))
    return all_reads

@profile
def create_labels(name):
    n_cluster = len(actual_labels[name])
    labels = []
    for idx, value in enumerate(actual_labels[name]):
        labels += [idx for i in range(1, value + 1)]
    return labels, n_cluster
