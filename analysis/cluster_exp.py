"""
File: synt.py
Coding: UTF-8
Author: lakj
Indentation : 4spaces
"""
from sensitivity_analysis import SensitivityAnalysis
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import process_datasets as pd
import putils as pu
from scipy import ndimage
import time
import os.path


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



################ Main script code ####################

# Algorithm parameters
n = 4000
repetitions = 1
refinement = 'indeg_guided'
write_csv = True
fix_bounds =  [0.05, 0.5]
tries = 20
fsize = 23

# Imbalancing of clusters
imbalanced = False
imb_cluster = [int(n*0.45), int(n*0.2), int(n*0.2), int(n*0.15)]
num_cs = [2,4,8,10,16,20,40]

header = "n,imbalanced,num_c,density,k,epsilon,sze_idx,nirr,refinement,desired_th,tpartition,treconstruction,tfiltering,tunweight,kvs_sze,kvs_fsze,l2_sze_G,l2_fsze_G,l1_sze_G,l1_fsze_G,min_l2_usze\n"
CSV_PATH = f"./data_unique_run/csv/cluster_exp/{n}.csv"
print(CSV_PATH)
if not os.path.isfile(CSV_PATH) and write_csv:
    with open(CSV_PATH, 'w') as f:
        f.write(header)

for repetition in range(0,repetitions):

    for num_c in num_cs:

        print(pu.to_header(f"r:{repetition+1}/{repetitions} n:{n} intra:0 inter:0"))

        # Graph creation
        if imbalanced:
            clusters = imb_cluster
        else:
            nc = n // num_c
            clusters = [nc]*num_c

        G, GT, labels = pd.custom_cluster_matrix(n, clusters, 0, 0, 0.05, 0)

        #G, GT, labels = pd.custom_crazy_cluster_matrix(n, clusters, internoise_lvl, inter_v, intranoise_lvl, 0)
        #G, GT, labels = pd.custom_cluster_matrix(n, [n], 1, 1, 0.1, 0)
        #labels = dominant_sets(G).astype('int64')

        data = {}
        data['G'] = G
        data['GT'] = GT
        data['bounds'] = fix_bounds
        data['labels'] = labels

        s = SensitivityAnalysis(data, refinement)
        s.verbose = True
        s.is_weighted = True
        #s.indensity_preservation = True


        if imbalanced:
            s.fast_search = False
        else:
            s.fast_search = True
        #s.drop_edges_between_irregular_pairs = False
        s.tries = tries

        tm = time.time()

        density = pd.density(G, weighted=True)
        print(pu.status(f"Density of G {density}"))

        print("[+] Finding bounds ...")
        bounds = s.find_bounds()
        print(pu.status(f"Bounds : {bounds}",'i'))

        print(pu.status(f"Finding partitions {refinement}..."))
        partitions = s.find_partitions()

        tpartition = time.time() - tm
        print(f"[T] Partitions found: {tpartition}")

        if partitions == {}:
            print(pu.status(f"No partition found", 'x'))
        else:
            n_partitions = len(partitions.keys()) 
            print(pu.status(f"{n_partitions} partitions found", 'i'))
            k = best_partition(partitions)

            epsilon = partitions[k][0]
            classes = partitions[k][1]
            sze_idx = partitions[k][2]
            reg_list = partitions[k][3]
            nirr = partitions[k][4]
            print(f"[+] Best partition - k:{k} epsilon:{epsilon:.4f} sze_idx:{sze_idx:.4f} irr_pairs:{nirr}")


        print("[+] Reconstruction t:0 --> SZE")
        sze = s.reconstruct_mat(0, classes, k, reg_list)
        treconstruction = time.time() - tm
        print(f"[T] Reconstruction: {treconstruction}")

        l2_sze_G = s.L2_metric(sze)
        l1_sze_G = s.L1_metric(sze)
        kvs_sze = s.KVS_metric(sze)

        print("[+] Filtering SZE --> FSZE")
        fsze = ndimage.median_filter(sze,fsize)
        tfiltering = time.time() - tm
        print(f"[T] Filtering: {tfiltering}")

        l2_fsze_G = s.L2_metric(fsze)
        l1_fsze_G = s.L1_metric(fsze)
        kvs_fsze = s.KVS_metric(fsze)

        print(f"[+] Unweighting SZE matrix by minimizing L2")
        min_l2_usze = 100
        desired_th = -1
        for i in np.arange(0,1,0.01):
            usze = (sze>(i*np.max(sze)))

            l2_usze = np.linalg.norm(G-usze)/G.shape[0]

            if l2_usze < min_l2_usze:
                desired_th = i
                min_l2_usze = l2_usze
                musze = usze

        tunweight = time.time() - tm
        print(f"[T] Unweight: {tunweight}")

        pu.plot_graphs([G, sze, fsze, musze], ["G", f"sze {k}", "fsze", "musze", f"n:{n} d:{density:.4f} num_c:{num_c}"], FILENAME=f"./data_unique_run/csv/cluster_exp/imgs/{n}_{num_c:03}.png", save=True, show=False)

        row = f"{n},{imbalanced},{num_c},{density:.4f},{k},{epsilon:.6f},{sze_idx:.4f},{nirr},{refinement},{desired_th:.4f},{tpartition:.2f},{treconstruction:.2f},{tfiltering:.2f},{tunweight:.2f},{kvs_sze:.4f},{kvs_fsze:.4f},{l2_sze_G:.4f},{l2_fsze_G:.4f},{l1_sze_G:.4f},{l1_fsze_G:.4f},{min_l2_usze:.4f}\n"
        print(row)
        if write_csv:
            with open(CSV_PATH, 'a') as f:
                f.write(row)


elapsed = time.time() - tm
print(f"[i] Time elapsed: {elapsed}")

ipdb.set_trace()


