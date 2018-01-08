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

def synthethic_cheese_graph(n):
    """ Generate a n,n cheese-style (with holes) graph of a given density d
    :param d: float density of the graph
    :return: np.array((n, n), dtype='float32') graph with density d
    """

    l = [x for x in range(0,n+1)]

    groups = zip(l[::5], l[1::5], l[2::5], l[3::5], l[4::5])
    groups = list(groups)

    a = np.array(groups[::2]).flatten()
    b = np.array(groups[1::2]).flatten()

    mat = np.ones((n,n))

    mat[np.ix_(a, b)] = 0
    mat += mat.T
    mat[np.where(mat == 1)] = 0
    mat /= 2

    np.fill_diagonal(mat, 0)

    return mat
    #print(mat)
    #plt.imshow(mat)
    #plt.show()


def density(G):
    """ Check density of a synthetic graph
    :param G: np.array((n, n), dtype='int8') graph
    :return: float density of the graph
    """
    n = G.shape[0]
    e = np.where(G == 1)[0].size / 2
    return e / ((n*(n-1))/2)

def best_partition(keci):
    """ Selects the best partition (highest sze_idx)
    """
    max_idx = -1
    max_k = -1

    for k in keci.keys():
        if keci[k][2] > max_idx:
            max_k = k
            max_idx = keci[k][2]

    return max_k

def plot_graphs(graph, sze, k):
    """ Plot the original graph vs sze_rec side by side
    """

    plt.subplot(1, 2, 1)
    plt.imshow(graph)
    plt.title("G")

    plt.subplot(1, 2, 2)
    plt.imshow(sze)
    plt.title(f"sze_rec, k={k}")

    plt.show()


######################################################
################ Main script code ####################
######################################################

### 1. Fix density d and size of the graph n ###
n = 6000
d = 0.85
refinement_type = 'indeg_guided' # or 'degree_based'

G = synthetic_graph(n, d)
check_validity(G, n, d)

#G = np.dot(G,G)
#G = synthethic_cheese_graph(n)
#d = density(G)

### 2. Prepares structure for the sensitivity analysis ###
data = {}
data['G'] = G

# Not useful with synthetic graphs
data['GT'] = []
data['bounds'] = []
data['labels'] = []

### 3. Instantiate the analysis class ###
s = SensitivityAnalysis(data, refinement_type)

s.verbose = True
s.drop_edges_between_irregular_pairs = True
s.indensity_preservation = True

### 4. Find bounds ###
print(f"[+] Finding bounds ...")
bounds = s.find_bounds()

### 5. Find partitions inside bounds ###
print(f"[+] Finding partitions ...eps k regularity sze_idx")
keci = s.find_partitions()

if keci == {}:
    print(f"[x] No partition found")

else:
    print(f"[+] {len(keci.keys())} partitions found")

    #max_k = best_partition(keci)
    #print(f"[+] Partition with the highest sze_idx k: {max_k} idx: {keci[max_k][2]:.4f}")

    ### 7. Recunstruction ###
    print(f"[+] Reconstruction and plot for each unique partition")

    # Good practice
    threshold = d - 0.03

    # Precalculation
    G = G.astype("float64")
    G_2 = G @ G
    G_3 = G_2 @ G
    G_n_triangles = np.trace(G_3) / 6.0

    for k in keci.keys():
        print(f"[*] Partition k:{k}")
        classes = keci[k][1]

        #s.thresholds_analysis(classes, k, np.arange(0.25, 0.45, 0.02), s.L2_metric)
        print(f"    Reconstruction with threshold: {threshold}")
        sze_rec = s.reconstruct_mat(threshold, classes, k, keci[k][3])

        # Count the number of triangles
        # stackoverflow.com/questions/39602404/numpy-matrix-exponentiation-gives-negative-value
        sze_rec = sze_rec.astype("float64")

        sze_2 = sze_rec @ sze_rec
        sze_3 = sze_2 @ sze_rec

        sze_n_triangles = np.trace(sze_3) / 6.0

        tott = sze_n_triangles / G_n_triangles
        print(f"    Triangles  sze: {sze_n_triangles}   G: {G_n_triangles} sze/G: {tott:.2f}")

        # Plot degree distribution
        plt.plot(sorted(np.diagonal(sze_2)), label=f"sze_rec k={k}")
        plt.plot(sorted(np.diagonal(G_2)), label="G")
        plt.title("Degree distribution")
        plt.legend()
        plt.show()

        ### To show the G vs. sze_rec ###
        plot_graphs(G, sze_rec, k)

        ### 8. Calculate distances ###
        dist1 = s.L1_metric(sze_rec)
        dist2 = s.L2_metric(sze_rec)
        print(f"    Distance: L1:{dist1:.4f}  L2:{dist2:.4f}")

