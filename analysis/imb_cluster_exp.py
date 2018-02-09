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
from scipy import ndimage
import time
import os.path


def best_threshold(mat, GT):

    print(f"[+] Unweighting...minimizing L2")
    min_l2 = 100
    threshold = -1
    for t in np.arange(0,1,0.01):
        umat = (mat>(t*np.max(mat)))

        l2 = np.linalg.norm(GT-umat)/GT.shape[0]

        if l2 < min_l2:
            threshold = t
            min_l2 = l2
            bestmat = umat

    return bestmat, min_l2, threshold


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

def random_clusters(n, num_c):

    imb_cluster = np.random.randint(1,num_c+1, size=(num_c,))
    imb_cluster = imb_cluster / imb_cluster.sum()
    imb_cluster = [int(n*dim) for dim in imb_cluster]
    if sum(imb_cluster) != n:
        imb_cluster[-1] = imb_cluster[-1]+(n-sum(imb_cluster))
    return imb_cluster 

################ Main script code ####################

# Algorithm parameters
n = 400
repetitions = 1
refinement = 'indeg_guided'
write_csv = True
fix_bounds =  [0.05, 0.5]
tries = 20
fsize = 23

internoiselvl = 0.2
intranoiselvl = 0

# Imbalancing of clusters
imbalanced = True
num_cs = [2,4,8,10,16,20,40]

header = "n,imbalanced,num_c,internoise,intranoise,density,k,epsilon,sze_idx,nirr,refinement,desired_th,tpartition,treconstruction,tfiltering,kvs_sze,kvs_fsze,l2_sze_G,l2_fsze_G,l1_sze_G,l1_fsze_G,l2_usze_GT,th_usze_GT,l2_ufsze_GT,th_ufsze_GT,l2_usze_G,th_usze_G,l2_ufsze_G,th_ufsze_G\n"
CSV_PATH = f"./data_unique_run/csv/imb_cluster_exp/{n}.csv"
print(CSV_PATH)
if not os.path.isfile(CSV_PATH) and write_csv:
    with open(CSV_PATH, 'w') as f:
        f.write(header)

for repetition in range(0,repetitions):
    
    for internoiselvl in [0, 0.05, 0.1, 0.2, 0.3]:

        for intranoiselvl in [0, 0.05, 0.1, 0.2, 0.3]:

            for num_c in num_cs:

                imb_cluster = random_clusters(n, num_c)

                # Graph creation
                if imbalanced:
                    clusters = imb_cluster
                else:
                    nc = n // num_c
                    clusters = [nc]*num_c

                print(pu.to_header(f"r:{repetition+1}/{repetitions} n:{n} num_c:{num_c} inter:{internoiselvl} intra:{intranoiselvl}"))
                G, GT, labels = pd.custom_cluster_matrix(n, clusters, internoiselvl, 1, intranoiselvl, 0)
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
                s.fast_search = False
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

                print(f"[+] Optimal Thresholds")
                usze_GT, l2_usze_GT, th_usze_GT = best_threshold(sze, GT)
                ufsze_GT, l2_ufsze_GT, th_ufsze_GT = best_threshold(fsze, GT)

                usze_G, l2_usze_G, th_usze_G = best_threshold(sze, G)
                ufsze_G, l2_ufsze_G, th_ufsze_G = best_threshold(fsze, G)
                #ipdb.set_trace()

                #pu.plot_graphs([G, sze, fsze, ufsze_GT], ["G", f"sze {k}", "fsze", "ufsze-GT", f"n:{n} d:{density:.4f} num_c:{num_c}"], FILENAME=f"./data_unique_run/csv/imb_cluster_exp/imgs/inter{internoiselvl:.2f}/{n}_{num_c:03}.png", save=True, show=False)

                row = f"{n},{imbalanced},{num_c},{internoiselvl:.2f},{intranoiselvl:.2f},{density:.4f},{k},{epsilon:.6f},{sze_idx:.4f},{nirr},{refinement},{tpartition:.2f},{treconstruction:.2f},{tfiltering:.2f},{kvs_sze:.4f},{kvs_fsze:.4f},{l2_sze_G:.4f},{l2_fsze_G:.4f},{l1_sze_G:.4f},{l1_fsze_G:.4f},{l2_usze_GT:.4f},{th_usze_GT:.2f},{l2_usze_GT:.4f},{th_usze_GT:.2f},{l2_ufsze_GT:.4f},{th_ufsze_GT:.2f},{l2_usze_G:.4f},{th_usze_G:.2f},{l2_ufsze_G:.4f},{th_ufsze_G:.2f}\n"
                print(row)
                if write_csv:
                    with open(CSV_PATH, 'a') as f:
                        f.write(row)


elapsed = time.time() - tm
print(f"[i] Time elapsed: {elapsed}")

ipdb.set_trace()

