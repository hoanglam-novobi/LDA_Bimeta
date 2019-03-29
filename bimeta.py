import networkx as nx
import random

def findNeighbors(graph_object, group):
    """
    :param group: list of integer
    """
    neighbor_hoods = []
    for node in group:
        neighbor_hoods += list(graph_object.neighbors(node))

    neighbor_hoods = list(set(neighbor_hoods) - set(group))
    return neighbor_hoods


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


