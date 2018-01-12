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
n = 2000
internoise = np.arange(0, 1, 0.03) 
internoise = [0.05]

l2_GT = []
l2_G = []
ari = []
kld = []
kld_GT = []


for intranoise_lvl in [0]:

    for internoise_lvl in internoise: #[0.05]
        print(internoise_lvl)

        #clusters = [int(n*0.7), int(n*0.2), int(n*0.05), int(n*0.05)]
        nc = n // 4
        clusters = [nc]*4

        G, GT, labels = pd.custom_cluster_matrix(n, clusters, internoise_lvl, 1, intranoise_lvl, 0)
        #G, GT, labels = pd.custom_crazy_cluster_matrix(n, clusters, internoise_lvl, 0.8, intranoise_lvl, 0)


        data = {}
        data['G'] = G
        data['GT'] = GT
        data['bounds'] = [0.01, 0.5]# [0.00018399953842163086, 0.3768310546875]
        data['labels'] = labels

        s = SensitivityAnalysis(data, 'indeg_guided')
        s.verbose = True
        s.is_weighted = True
        #s.drop_edges_between_irregular_pairs = False

        t = time.time()
        print("running...")
        print(pd.density(G, weighted=True))
        regular, k, classes, sze_idx, reg_list , nirr = s.run_alg(0.33)
        elapsed = time.time() - t
        print(f"{regular}, {k}, {sze_idx}, {nirr}")
        print(elapsed)


        for t in np.arange(0,1, 0.05):
            sze_rec = s.reconstruct_mat(t, classes, k, reg_list)

            print(s.L2_metric_GT(sze_rec))
            print(s.L2_metric(sze_rec))
            print(s.KVS_metric(sze_rec))
            pu.plot_graphs([G, sze_rec], ["G", f"sze_rec {k}"])

        #ipdb.set_trace()

        sys.exit()

        print("[+] Finding bounds ...")
        bounds = s.find_bounds()
        print(f"[i] Bounds : {bounds}")

        print(f"[+] Finding partitions ...")
        partitions = s.find_partitions()

        if partitions == {}:
            print("[x] No partition found")
        else:
            n_partitions = len(partitions.keys()) 
            print(f"[i] {n_partitions} partitions found")
            k = best_partition(partitions)

            epsilon = partitions[k][0]
            classes = partitions[k][1]
            sze_idx = partitions[k][2]
            reg_list = partitions[k][3]
            nirr = partitions[k][4]
            print(f"[+] Best partition - k:{k} epsilon:{epsilon:.4f} sze_idx:{sze_idx:.4f} irr_pairs:{nirr}")

            print("[+] Reconstruction and plot for each unique partition")

            #d = pd.density(G) + 0.03
            #for t in np.arange(0.1, 1, 0.05):
            sze_rec = s.reconstruct_mat(0, classes, k, reg_list)

            #pu.plot_graphs([GT, G, sze_rec], ["GT", "G", f"sze_rec {k}"])

            l2_GT.append(s.L2_metric_GT(sze_rec))
            l2_G.append(s.L2_metric(sze_rec))
            ari.append(s.KVS_metric(sze_rec))
            #kld.append(s.KLdivergence_metric(sze_rec))
            #kld_GT.append(s.KLdivergence_metric_GT(sze_rec))

            #print(l2g_s)
            #print(l2gt_s)


# Plot
plt.plot(internoise, l2_GT, label="L2(REC, GT)")
plt.plot(internoise, l2_G, label="L2(REC, G)")
plt.title(f"N:{n}")
plt.ylabel('L2')
plt.xlabel('Internoise')
plt.grid()
plt.legend(loc='lower right')
plt.savefig(f"./imgs/{n}_l2.png")

# Plot
plt.figure()
plt.plot(internoise, ari)
plt.title(f"N:{n}")
plt.ylabel('Adj Rand Idx')
plt.xlabel('Internoise')
plt.grid()
plt.savefig(f"./imgs/{n}_ari.png")

