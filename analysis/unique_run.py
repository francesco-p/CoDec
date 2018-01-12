"""
File: synt.py
Coding: UTF-8
Author: lakj
Indentation : 4spaces
"""
from sensitivity_analysis import SensitivityAnalysis
import numpy as np
import matplotlib.pyplot as plt
#import ipdb
import process_datasets as pd
import putils as pu
import sys
import scipy.signal as sp
from scipy import ndimage
import time


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



######################################################
################ Main script code ####################
######################################################


#for internoise_lvl in [0.1, 0.3, 0.5, 0.7]:
n = 4000

internoise = [0.05] #np.arange(0, 1, 0.03) 
intranoise = [0.1]

for intranoise_lvl in intranoise:

    for internoise_lvl in internoise:

        #clusters = [int(n*0.7), int(n*0.2), int(n*0.05), int(n*0.05)]
        nc = n // 4
        clusters = [nc]*4

        print(f"[+] Generating graph with inter:{internoise_lvl} intra:{intranoise_lvl}")
        G, GT, labels = pd.custom_cluster_matrix(n, clusters, internoise_lvl, 1, intranoise_lvl, 0)
        #G, GT, labels = pd.custom_crazy_cluster_matrix(n, clusters, internoise_lvl, 0.8, intranoise_lvl, 0)

        data = {}
        data['G'] = G
        data['GT'] = GT
        data['bounds'] = [0.01, 0.5]
        data['labels'] = labels

        s = SensitivityAnalysis(data, 'indeg_guided')
        s.verbose = True
        s.is_weighted = True
        #s.drop_edges_between_irregular_pairs = False

        t = time.time()
        print("[+] Density of G {0}".format(pd.density(G, weighted=True)))
        print("[+] Running...")
        regular, k, classes, sze_idx, reg_list , nirr = s.run_alg(0.33) #0.25 n20000
        elapsed = time.time() - t
        print(f"[+] Partition found: {regular}, {k}, {sze_idx}, {nirr}")
        print(f"[i] Time elapsed: {elapsed}")



        print("[+] Reconstruction")
        for t in np.arange(0,1, 0.05):
            print(f"  Threshold: {t:.2f}")
            sze_rec = s.reconstruct_mat(t, classes, k, reg_list)
            f_sze_rec = ndimage.median_filter(sze_rec,15)
            print("    l2_gt   {0:.4f}".format(s.L2_metric_GT(sze_rec)))
            print("    l2_g    {0:.4f}".format(s.L2_metric(sze_rec)))
            print("    adj_idx {0:.4f}".format(s.KVS_metric(sze_rec)))
            print("    f_l2_gt   {0:.4f}".format(s.L2_metric_GT(f_sze_rec)))
            print("    f_l2_g    {0:.4f}".format(s.L2_metric(f_sze_rec)))
            print("    f_adj_idx {0:.4f}".format(s.KVS_metric(f_sze_rec)))
            pu.plot_graphs([G, sze_rec, f_sze_rec], ["G", f"sze_rec {k}", "filtered sze"])

        #ipdb.set_trace()

