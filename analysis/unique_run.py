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
import sys
import scipy.signal as sp
import io 
from scipy import ndimage
from math import ceil
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



######################################################
################ Main script code ####################
######################################################



repetitions = 1
n = 200
refinement = 'indeg_guided'

# Imbalancing of clusters
inbalanced = False
imb_cluster = [int(n*0.45), int(n*0.2), int(n*0.2), int(n*0.15)]
num_c = 4

write_csv = False

fix_bounds =  [0.05, 0.5]

intranoise = [0]
internoise = [0.2, 0.4, 0.5, 0.6, 0.8]
inter_v = 1
intra_v = 0

header = f"n,graph_name,inbalanced,internoise,intranoise,density,k,epsilon,sze_idx,nirr,refinement,best_threshold,l2_G_GT,l1_G_GT,kvs_G,l2_fG_GT,l1_fG_GT,kvs_fG,l2_sze_GT,l1_sze_GT,kvs_sze,l2_fsze_GT,l1_fsze_GT,kvs_fsze,DS_G,DS_fG,DS_sze,DS_fsze,l2_sze_G,l2_fsze_fG,l2_sze_fG,l2_fsze_G,l1_sze_G\n"

if intranoise[0] != 0:
    CSV_PATH = f"./data_unique_run/csv/{n}_intra.csv"
else:
    CSV_PATH = f"./data_unique_run/csv/{n}.csv"

print(CSV_PATH)


if not os.path.isfile(CSV_PATH) and write_csv:
    with io.open(CSV_PATH, 'w') as f:
        f.write(header)


idg = 0
for repetition in range(0,repetitions):


    for intranoise_lvl in intranoise:

        for internoise_lvl in internoise:

            print(pu.to_header(f"r:{repetition+1}/{repetitions} n:{n} intra:{intranoise_lvl}/{intranoise} inter:{internoise_lvl}/{internoise}"))

            if inbalanced:
                clusters = imb_cluster
            else:
                nc = n // num_c
                clusters = [nc]*num_c

            G, GT, labels = pd.custom_cluster_matrix(n, clusters, internoise_lvl, inter_v, intranoise_lvl, intra_v)
            #G, GT, labels = pd.custom_crazy_cluster_matrix(n, clusters, internoise_lvl, inter_v, intranoise_lvl, 0)
            #G, GT, labels = pd.custom_cluster_matrix(n, [n], 1, 1, 0.1, 0)
            #labels = dominant_sets(G).astype('int64')


            density = pd.density(G, weighted=True)

            data = {}
            data['G'] = G
            data['GT'] = GT
            data['bounds'] = fix_bounds
            data['labels'] = labels

            s = SensitivityAnalysis(data, refinement)
            s.verbose = True
            s.is_weighted = True

            if inbalanced:
                s.fast_search = False
            else:
                s.fast_search = True

            #s.drop_edges_between_irregular_pairs = False

            tm = time.time()

            print(pu.status(f"Density of G {density}"))

            print("[+] Finding bounds ...")
            bounds = s.find_bounds()
            print(pu.status(f"Bounds : {bounds}",'i'))

            print(pu.status(f"Finding partitions {refinement}..."))
            partitions = s.find_partitions()

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

            #print("[+] Running...")
            #regular, k, classes, sze_idx, reg_list , nirr = s.run_alg(0.31) #0.25 n20000
            #print(f"[+] Partition found: {regular}, {k}, {sze_idx}, {nirr}")

            l2_G_GT = s.L2_metric_GT(G)
            l1_G_GT = s.L1_metric_GT(G)
            kvs_G = s.KVS_metric(G)
            DS_G = s.DS_metric(G)

            print("[+] Filtering G")
            fG = ndimage.median_filter(G,23)
            l2_fG_GT = s.L2_metric_GT(fG)
            l1_fG_GT = s.L1_metric_GT(fG)
            kvs_fG = s.KVS_metric(fG)
            DS_fG = s.DS_metric(fG)


            print("[+] Reconstruction")
            measures = [-1]*4
            #for t in [0, 0.2, 0.4, 0.6, density+0.05, density+0.03]: #np.arange(0,1, 0.05):
            for t in [0]: #np.arange(0,1, 0.05): # CHEATING WE DONT KNOW GT
                print(f"  Threshold: {t:.2f}")
                sze = s.reconstruct_mat(t, classes, k, reg_list)

                l2_sze_GT = s.L2_metric_GT(sze)
                l1_sze_GT = s.L1_metric_GT(sze)
                kvs_sze = s.KVS_metric(sze)

                DS_sze = s.DS_metric(sze)

                print("[+] Filtering sze")
                fsze = ndimage.median_filter(sze,23)

                l2_fsze_GT = s.L2_metric_GT(fsze)
                l1_fsze_GT = s.L1_metric_GT(fsze)
                kvs_fsze = s.KVS_metric(fsze)

                DS_fsze = s.DS_metric(fsze)


                if kvs_fsze > measures[0]:
                    measures = [kvs_fsze, l2_fsze_GT, l1_fsze_GT, t]


                """
                print(f"[+] Remove weights with thresholds")
                min_l2_usze_GT = 100
                desired_th = -1
                for i in np.arange(0,1,0.005):
                    usze = (sze>(i*np.max(sze)))
                    l2_usze_GT = np.linalg.norm(GT-usze)/GT.shape[0]
                    pu.cprint(f"{l2_usze_GT}", COL=pu.BLU)

                    density_usze = pd.density(usze, weighted=False)

                    if l2_usze_GT < min_l2_usze_GT:
                        desired_th = i
                        min_l2_usze_GT = l2_usze_GT
                        mufsze = usze
                    print("{:.4f} : {:.4f} : {:.4f} : {:.4f}".format(i, density_usze, density, l2_usze_GT))


                """
                print(f"[+] Remove weights with thresholds")
                min_l2_usze_GT = 100
                desired_th = -1
                for i in np.arange(0,1,0.005):
                    ufsze = (fsze>(i*np.max(fsze)))
                    l2_usze_GT = np.linalg.norm(GT-ufsze)/GT.shape[0]

                    density_usze = pd.density(ufsze, weighted=False)

                    if l2_usze_GT < min_l2_usze_GT:
                        desired_th = i
                        min_l2_usze_GT = l2_usze_GT
                        mufsze = ufsze

                    print("{:.4f} : {:.4f} : {:.4f} : {:.4f}".format(i, density_usze, density, l2_usze_GT))


                pu.plot_graphs([G, sze, fsze, fG, mufsze ], [f"G {n} {internoise_lvl:.2f}", f"sze {k}", "filtered sze", "filtered G", f"ufsze {desired_th:.4f}"])

                if write_csv:
                    with open(f"./data_unique_run/csv/tmp/{n}_{internoise_lvl:.2f}.csv", 'a') as f:
                        f.write("{:.4f},{:.4f},{:.4f},{:.4f}\n".format(density,desired_th, min_l2_usze_GT, np.linalg.norm(GT-G)/GT.shape[0]))





            kvs_fsze = measures[0]
            l2_fsze_GT = measures[1]
            l1_fsze_GT = measures[2]
            best_threshold = measures[3]

            l2_sze_G = s.L2_metric(sze)
            l1_sze_G = s.L1_metric(sze)
            l2_fsze_fG = np.linalg.norm(fG-fsze)/fG.shape[0]
            l2_sze_fG = np.linalg.norm(fG-sze)/fG.shape[0]
            l2_fsze_G = np.linalg.norm(G-fsze)/G.shape[0]

            row = f"{n},{idg},{inbalanced},{internoise_lvl:.2f},{intranoise_lvl:.2f},{density:.4f},{k},{epsilon:.10f},{sze_idx:.5f},{nirr},{refinement},{best_threshold:.2f},{l2_G_GT:.4f},{l1_G_GT:.4f},{kvs_G:.4f},{l2_fG_GT:.4f},{l1_fG_GT:.4f},{kvs_fG:.4f},{l2_sze_GT:.4f},{l1_sze_GT:.4f},{kvs_sze:.4f},{l2_fsze_GT:.4f},{l1_fsze_GT:.4f},{kvs_fsze:.4f},{DS_G:.4f},{DS_fG:.4f},{DS_sze:.4f},{DS_fsze:.4f},{l2_sze_G:.4f},{l2_fsze_fG:.4f},{l2_sze_fG:.4f},{l2_fsze_G:.4f},{l1_sze_G:.4f}\n"

            print(row, end="")
            if write_csv:
                with io.open(CSV_PATH, 'a') as f:
                    f.write(row)

            idg += 1


elapsed = time.time() - tm
print(f"[i] Time elapsed: {elapsed}")

ipdb.set_trace()

