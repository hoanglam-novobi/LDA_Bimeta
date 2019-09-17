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
    'S7': [141928, 75183, 126183, 588088, 722168],    
    'S8': [99577, 82961, 37493, 73514, 162679],
    'S9': [103440, 124564, 47629, 60623, 116341, 167267, 216912, 82012, 238138, 141070, 182576, 177691, 279117, 115226, 181562],
    'S10': [62115, 99266, 109121, 124771, 131964, 159574, 194199, 72298, 33815, 34304, 156301, 106183, 82166, 193719, 178956, 100570, 189889, 286620, 190621, 190397, 259854, 240963, 100830, 261254, 305786, 167722, 228613, 268766, 353996, 105999],
    'L1': [93808, 82880],
    'L2': [93505, 166063],
    'L3': [93789, 248659],
    'L4': [94031, 331297],
    'L5': [93796, 414413],
    'L6': [93361, 497728]
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
        labels += [idx]*value
    return labels, n_cluster
