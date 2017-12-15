"""
File: single_run.py
Description:
 1. Fix density d and size of the graph n
 2. Prepares structure for the sensitivity analysis
 3. Instantiate the analysis class
 4. Find bounds
 5. Find partitions inside bounds
 6. Select the partition with the highest sze_idx
 7. Recunstruction
    - plot if you want
 8. Calculate a distance
Coding: UTF-8
Author: lakj
Indentation : 4spaces
"""
from sensitivity_analysis import SensitivityAnalysis
import numpy as np
import matplotlib.pyplot as plt
import ipdb

def check_validity(G, n, desired_d):
    """ Checks dimension and data type of the graph to reduce space consumption
    Checks that the density is in desired_density +/- tolerance
    """
    assert G.shape[0] == n, "Dimension is not correct"
    assert G.dtype == 'int8', "Data type of the graph is not correct"
    tolerance = 0.015
    desired_d = round(desired_d, 2)
    graph_d = round(density(G), 2)
    assert desired_d-tolerance < graph_d and graph_d < desired_d+tolerance, "Density is not correct"


def synthetic_graph(n, d):
    """ Generate a n,n graph with a given density d
    :param d: float density of the graph
    :return: np.array((n, n), dtype='float32') graph with density d
    """
    G = np.tril(np.random.random((n, n)) <= d, -1).astype('int8')
    return G + G.T


def density(G):
    """ Check density of a synthetic graph
    :param G: np.array((n, n), dtype='int8') graph
    :return: float density of the graph
    """
    n = G.shape[0]
    e = np.where(G == 1)[0].size / 2
    return e / ((n*(n-1))/2)

def plot_graphs(graph, sze):
    """ Plot graph vs sze side by side
    """

    plt.subplot(1, 2, 1)
    plt.imshow(graph)
    plt.title("G")

    plt.subplot(1, 2, 2)
    plt.imshow(sze)
    plt.title("sze_rec")

    plt.show()



######################################################
################ Main script code ####################
######################################################

### 1. Fix density d and size of the graph n ###
n = 600
d = 0.7

G = synthetic_graph(n, d)
check_validity(G, n, d)

### 2. Prepares structure for the sensitivity analysis ###
data = {}
data['G'] = G
# Not useful with synthetic graphs
data['GT'] = []
data['bounds'] = []
data['labels'] = []

### 3. Instantiate the analysis class ###
s = SensitivityAnalysis(data, 'degree_based')
s.verbose = True

### 4. Find bounds ###
# Returns a list
#   bounds[0] = epsilon edge, bounds[1] = epsilon trivial
print(f"[+] Finding bounds ...")
bounds = s.find_bounds()

### 5. Find partitions inside bounds ###
# Returns a dictionary of 3-uples
#   {k: (epsilon, classes array, sze_idx), ...}
# ex:
#   {'128': (0.45735, np.array[1,3,2,3,4,1,1,2, ...], 0.253 ), '256':...}
print(f"[+] Finding partitions ...")
keci = s.find_partitions()

if keci == {}:
    print(f"[x] No partition found")

else:
    print(f"[+] {len(keci.keys())} partitions found")

    ### 6. Select the partition with the highest sze_idx ###
    max_idx = -1
    max_k = -1

    for k in keci.keys():
        if keci[k][2] > max_idx:
            max_k = k
            max_idx = keci[k][2]

    print(f"[+] Partition with the highest sze_idx k: {max_k} idx: {keci[max_k][2]:.4f}")

    ### 7. Recunstruction ###
    threshold = 0
    classes = keci[max_k][1]
    print(f"[+] Reconstruction with threshold: {threshold}")
    sze_rec = s.reconstruct_mat(threshold, classes, max_k)

    ### To show the G vs. sze_rec ###
    plot_graphs(G, sze_rec)

    ### 8. Calculate a distance ###
    dist = s.L2_metric(sze_rec)
    print(f"[+] Distance: {dist}")

