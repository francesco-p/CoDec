"""
File: synt.py
Coding: UTF-8
Author: lakj
Indentation : 4spaces
"""
from codec import Codec
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import process_datasets as pd
import putils as pu
import time
import os.path
import metrics
from stats import Stats


##############################################
############## Main script code ##############
##############################################

def unweight(fsze, graph):
    """ Unweighting of fsze matrix by minimizing l2 with respect to graph
    :param fsze: np.array((n, n)) matrix to be unweighted
    :paragm graph: np.array((n, n)) target matrix
    :return: np.array((n, n)), float, float unweighted matrix, l2 distance, value of the threshold
    """
    print(f"[+] Unweighting...")
    min_l2 = 100
    threshold = -1
    for t in np.arange(0,1,0.01):
        umat = (fsze>(t*np.max(fsze)))

        l2 = np.linalg.norm(graph-umat)/graph.shape[0]

        if l2 < min_l2:
            threshold = t
            min_l2 = l2
            bestmat = umat

    return bestmat, min_l2, threshold

tm = time.time()

refinement = 'indeg_guided'
ksize = 23
imbalanced = False
num_c = 8
internoiselvl = 0
intranoiselvl = 0

clusters = pd.random_clusters(2000, 8)
G, GT, labeling = pd.custom_cluster_matrix(2000, clusters, 0.1, 1, 0.2, 0)

s = Stats("/tmp/test.csv")
c = Codec(0, 0.5, 20)

k, epsilon, classes, sze_idx, reg_list, nirr = c.compress(G, refinement)
tcompression = time.time() - tm

sze = c.decompress(G, 0, classes, k, reg_list)
tdecompression = time.time() - tm

fsze = c.post_decompression(sze, ksize)
tpostdecompression = time.time() - tm

red = c.reduced_matrix(G, k, epsilon, classes, reg_list)

telapsed = [tcompression, tdecompression-tcompression, tpostdecompression-tdecompression]

ufsze, x, y = unweight(fsze, G)

pu.plot_graphs([G, sze, fsze, ufsze], ["G", f"sze:{k}", "fsze", "ufsze"])


#s.synth_stats(imbalanced, num_c, internoiselvl, intranoiselvl, k, epsilon, sze_idx, nirr, refinement, G, GT, labeling, sze, fsze, red, telapsed, write=False, plot=True, pp=True)

#print(metrics.spectral_dist(G, red))
