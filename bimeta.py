import networkx as nx
import random
import logging
import time
import numpy as np

# CONFIG FOR LOGGING MEMORY_PROFILER
import sys
from memory_profiler import profile, LogFile

sys.stdout = LogFile(__name__)


@profile
def findNeighbors(graph_object, group):
    """
    :param group: list of integer
    """
    neighbor_hoods = []
    for node in group:
        neighbor_hoods += list(graph_object.neighbors(node))

    neighbor_hoods = list(set(neighbor_hoods) - set(group))
    return neighbor_hoods


@profile
def phase_1_Bimeta(graph_object):
    maxS = 9000
    groups = {}
    # get all nodes from the graph (V_temp = V)
    all_nodes = list(graph_object.nodes())
    n_nodes = len(all_nodes)
    print("Total nodes: ", n_nodes)
    print("Nodes: ", all_nodes)
    i = 1
    while all_nodes:
        # create new group Gi = {S(Gi), NS(Gi)}
        S = []
        NS = []
        # random choose v from V_temp
        v = random.choice(all_nodes)
        S.append(v)
        print("Random choose v: ", v)
        # remove v from V_temp
        all_nodes.remove(v)
        n_S = len(S)
        neighbors = findNeighbors(graph_object, S + NS)
        print("Neighbors of v: ", neighbors)
        while n_S <= maxS and neighbors:
            node = random.choice(neighbors)
            if node in all_nodes:
                print("Remove node %d." % node)
                all_nodes.remove(node)

            if node not in findNeighbors(graph_object, S):
                print("Add node %d in S of group %d" % (node, i))
                S.append(node)
            else:
                print("Add node %d in NS of group %d" % (node, i))
                NS.append(node)

            n_S = len(S)
            neighbors = findNeighbors(graph_object, S + NS)

        groups[i] = {'S': S, 'NS': NS}
        i += 1
    return groups


@profile
def read_bimeta_input(path, file_name):
    """
    Read the result of phase 1 from Bimeta
    The id of read in the file is from 1 so we need to convert the id from 0 to n-1.
    :param path:
    :param file_name:
    :return:
    """
    t1 = time.time()
    logging.info("Reading seeds group ...")
    res = {}
    with open(path + file_name, 'r') as f:
        for row_id, line in enumerate(f):
            line = line.split(',')
            if len(line) <= 1:
                continue
            res[row_id] = [int(read_id) - 1 for read_id in line[1:]]
    t2 = time.time()
    logging.info("Finished read the result from phase 1 Bimeta in %f (s)." % (t2 - t1))
    return res


@profile
def create_characteristic_vector(top_dist, seeds):
    t1 = time.time()
    logging.info("Creating median vector in seeds.")
    res = []
    for id, value in seeds.items():
        tmp = top_dist[value, :]
        res.append(np.mean(tmp, axis=0))
    t2 = time.time()
    logging.info("Finished creating median vector of small group in seeds in %f (s)." % (t2 - t1))
    return np.array(res)
