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
ksize = 23

vals = [0.1 , 0.2 , 0.5]
num_cs = [4,8,20]

# Query graph, the bigger one
clusters = pd.random_clusters(1000, 8)
G, GT, labeling = pd.custom_cluster_matrix(1000, clusters, 0.1, 1, 0.1, 0)
c = Codec(0.15, 0.5, 5)
k, epsilon, classes, sze_idx, reg_list, nirr = c.compress(G, refinement)
query = c.reduced_matrix(G, k, epsilon, classes, reg_list)

# Database
c = Codec(0.2, 0.5, 5)
num_c = 8
n = 2000
for r in range(20):
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

        #row = f"{density:.4f},{inter:.2f},{intra:.2f},{num_c:02d},{sd:.4f}\n"
        row = f"{density:.4f},{i:.2f},{i:.2f},{num_c:02d},{sd:.4f}\n"
        with open("/tmp/stats.csv", 'a') as f:
            f.write(row)


#ipdb.set_trace()
