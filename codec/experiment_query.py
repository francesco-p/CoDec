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
from tqdm import tqdm


##############################################
############## Main script code ##############
##############################################


refinement = 'indeg_guided'

vals = [0.1 , 0.2 , 0.5, 0.7]
first = True
n = 1000

# Query graph, the bigger one
clusters = pd.random_clusters(n, 8)
inter = 0.1
intra = 0.1
oG, oGT, olabeling = pd.custom_cluster_matrix(n, clusters, inter, 1, intra, 0)
c = Codec(0.15, 0.5, 5)
k, epsilon, classes, sze_idx, reg_list, nirr = c.compress(oG, refinement)
query = c.reduced_matrix(oG, k, epsilon, classes, reg_list)



# Database
c = Codec(0.2, 0.5, 5)
num_c = 8
for r in range(5):
    for i in vals:

        clusters = pd.random_clusters(n, num_c)
        G, GT, labeling = pd.custom_cluster_matrix(n, clusters, i, 1, i, 0)
        k, epsilon, classes, sze_idx, reg_list, nirr = c.compress(G, refinement)
        red = c.reduced_matrix(G, k, epsilon, classes, reg_list)

        if query.shape[0] > red.shape[0]:
            sd = metrics.spectral_dist(query, red)
        else:
            sd = metrics.spectral_dist(red,query)

        density = pd.density(red)

        osd = metrics.spectral_dist(oG, G)

        if first:
            row = f"# Query: {inter:.2f},{intra:.2f}\n"
            first = False
        else:
            row = f"{density:.4f},{i:.2f},{i:.2f},{num_c:02d},{sd:.4f},{osd:.4f}\n"

        with open("/tmp/stats.csv", 'a') as f:
            f.write(row)


#ipdb.set_trace()
