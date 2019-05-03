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
            if line.rstrip():
                line = line.rstrip().split(',')
                if len(line) < 1:
                    continue
                res[row_id] = [int(read_id) - 1 for read_id in line]
            else:
                res[row_id] = []
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


def assign_cluster(mean_vector_clusters, seeds):
    """
    Assign the cluster of the mean vector of a group in seed to each read in seed
    :param mean_vector_clusters: cluster of the mean vector of a group in seeds
    :type mean_vector_clusters: list of integer
    :param seeds: reads in a group in seeds
    :type seeds: a dictionary
    :return:
    """
    logging.info("Assign cluster for reads in seeds.")
    t1 = time.time()
    # store cluster of each read
    result = {}
    for group_id, cluster in enumerate(mean_vector_clusters):
        for read_id in seeds.get(group_id, []):
            result[read_id] = cluster

    # sort dict by read_id
    sorted_dict = sorted(result.items(), key=lambda key: key[0])
    cluster_reads = [item[1] for item in sorted_dict]
    logging.info("Length result: ", len(result))
    t2 = time.time()
    logging.info("Finished assign cluster for reads in seeds in %f (s)." % (t2 - t1))
    return cluster_reads


def genOverlapGraph(reads, l=20, threshold=5):
    k_mer_dict = {}
    for idx, read in enumerate(reads):
        for i in range(0, len(read) - l + 1):
            k_mer = read[i:i + l]
            if k_mer not in k_mer_dict:
                k_mer_dict[k_mer] = [idx]
            else:
                k_mer_dict[k_mer].append(idx)

    for key, value in k_mer_dict.items():
        if len(value) >= 200:
            k_mer_dict.pop(key)

    overlap_dict = {}

    G = nx.Graph()
    for i in range(0, len(reads) - 1):
        for j in range(i + 1, len(reads)):
            if len(dict[i].intersection(dict[j])) >= threshold:
                G.add_edge(i + 1, j + 1)
        print("Testing from %d..." % (i))
    return G


def find_max_independent_set(graph_object):
    """
    Find list of nodes in independent set of the graph.
    Using an approximate algorithms in the library Networkx
    :param graph_object:
    :return: list of node_id from 1 to n
    """
    return nx.maximal_independent_set(graph_object)
