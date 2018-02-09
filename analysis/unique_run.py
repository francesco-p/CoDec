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
repetitions = 1
n = 1000
refinement = 'indeg_guided'
write_csv = False
fix_bounds =  [0.05, 0.5]
tries = 20
fsize = 23


# Imbalancing of clusters
imbalanced = False
num_c = 4

imb_cluster = [int(n*0.45), int(n*0.2), int(n*0.2), int(n*0.15)]


#intranoise = [0]
#internoise = [0.2, 0.4, 0.5, 0.6, 0.8]
intranoise = [0.1]
internoise = [0.1]#, 0.4, 0.6] #[0.2,0.3,0.4,0.5,0.6,0.8,0.9,0.95]

inter_v = 1
intra_v = 0


header = f"n,imbalanced,num_c,internoise_lvl,intranoise_lvl,density,k,epsilon,sze_idx,nirr,refinement,desired_th,tpartition,treconstruction,tfiltering,tunweight,l2_G_GT,l1_G_GT,kvs_G,l2_fG_GT,l1_fG_GT,kvs_fG,l2_sze_GT,l1_sze_GT,kvs_sze,l2_fsze_GT,l1_fsze_GT,kvs_fsze,l2_sze_G,l2_fsze_fG,l2_sze_fG,l2_fsze_G,l1_sze_G,min_l2_usze,l2_fmusze_GT,l1_fmusze_GT,kvs_fmusze\n"
CSV_PATH = f"./data_unique_run/csv/{n}.csv"
print(CSV_PATH)
if not os.path.isfile(CSV_PATH) and write_csv:
    with open(CSV_PATH, 'w') as f:
        f.write(header)


for repetition in range(0,repetitions):

    for intranoise_lvl in intranoise:

        for internoise_lvl in internoise:

            print(pu.to_header(f"r:{repetition+1}/{repetitions} n:{n} intra:{intranoise_lvl}/{intranoise} inter:{internoise_lvl}/{internoise}"))

            # Graph creation
            if imbalanced:
                clusters = imb_cluster
            else:
                nc = n // num_c
                clusters = [nc]*num_c

            G, GT, labels = pd.custom_cluster_matrix(n, clusters, internoise_lvl, inter_v, intranoise_lvl, intra_v)
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


            # Measuring stuff
            l2_G_GT = s.L2_metric_GT(G)
            l1_G_GT = s.L1_metric_GT(G)
            kvs_G = s.KVS_metric(G)
            #DS_G = s.DS_metric(G)
            print("[+] Filtering G")
            fG = ndimage.median_filter(G,fsize)
            l2_fG_GT = s.L2_metric_GT(fG)
            l1_fG_GT = s.L1_metric_GT(fG)
            kvs_fG = s.KVS_metric(fG)
            #DS_fG = s.DS_metric(fG)


            print("[+] Reconstruction t:0 --> SZE")
            sze = s.reconstruct_mat(0, classes, k, reg_list)
            treconstruction = time.time() - tm
            print(f"[T] Reconstruction: {treconstruction}")
            l2_sze_GT = s.L2_metric_GT(sze)
            l1_sze_GT = s.L1_metric_GT(sze)
            kvs_sze = s.KVS_metric(sze)
            #DS_sze = s.DS_metric(sze)


            print("[+] Filtering SZE --> FSZE")
            fsze = ndimage.median_filter(sze,fsize)
            tfiltering = time.time() - tm
            print(f"[T] Filtering: {tfiltering}")
            l2_fsze_GT = s.L2_metric_GT(fsze)
            l1_fsze_GT = s.L1_metric_GT(fsze)
            kvs_fsze = s.KVS_metric(fsze)
            #DS_fsze = s.DS_metric(fsze)


            print(f"[+] Unweighting SZE matrix by minimizing L2")
            min_l2_usze = 100
            desired_th = -1
            for i in np.arange(0,1,0.005):
                usze = (sze>(i*np.max(sze)))

                l2_usze = np.linalg.norm(GT-usze)/GT.shape[0]
                #l2_usze = np.linalg.norm(G-usze)/G.shape[0]

                if l2_usze < min_l2_usze:
                    desired_th = i
                    min_l2_usze = l2_usze
                    musze = usze

                density_usze = pd.density(usze, weighted=False)


            if write_csv:
                with open(f"./data_unique_run/csv/tmp/{n}.csv", 'a') as f:
                    f.write(f"{internoise_lvl:.4f},{intranoise_lvl:.4f},{density:.4f},{min_l2_usze:.4f}\n")

            tunweight = time.time() - tm
            print(f"[T] Unweight: {tunweight}")

            #fmusze = musze
            fmusze = ndimage.median_filter(musze,fsize)
            
            red = s.generate_reduced_sim_mat(k, epsilon, classes, reg_list)

            #pu.plot_graphs([G, sze, fsze, fG, fmusze, red], [f"G n:{n} {internoise_lvl:.2f}/{intranoise_lvl:.2f}", f"sze {k}", "filtered sze", "filtered G", f"fmusze {desired_th:.4f}", "red", "matrici"])
            #pu.plot_graphs([G, sze, fsze, red], [f"G n:{n} {internoise_lvl:.2f}/{intranoise_lvl:.2f}", f"sze {k}", "filtered sze", "red"])
            pu.plot_graphs([G, red, sze, fsze], [f"G", f"RED k:{k}", f"SZE k:{k}", "FSZE", "CoDec"])


            l2_sze_G = s.L2_metric(sze)
            l1_sze_G = s.L1_metric(sze)
            l2_fsze_fG = np.linalg.norm(fG-fsze)/fG.shape[0]
            l2_sze_fG = np.linalg.norm(fG-sze)/fG.shape[0]
            l2_fsze_G = np.linalg.norm(G-fsze)/G.shape[0]
 
            l2_fmusze_GT = s.L2_metric_GT(fmusze)
            l1_fmusze_GT = s.L1_metric_GT(fmusze)
            kvs_fmusze = s.KVS_metric(fmusze)


            row = f"{n},{imbalanced},{num_c},{internoise_lvl:.2f},{intranoise_lvl:.2f},{density:.4f},{k},{epsilon:.6f},{sze_idx:.4f},{nirr},{refinement},{desired_th:.4f},{tpartition:.2f},{treconstruction:.2f},{tfiltering:.2f},{tunweight:.2f},{l2_G_GT:.4f},{l1_G_GT:.4f},{kvs_G:.4f},{l2_fG_GT:.4f},{l1_fG_GT:.4f},{kvs_fG:.4f},{l2_sze_GT:.4f},{l1_sze_GT:.4f},{kvs_sze:.4f},{l2_fsze_GT:.4f},{l1_fsze_GT:.4f},{kvs_fsze:.4f},{l2_sze_G:.4f},{l2_fsze_fG:.4f},{l2_sze_fG:.4f},{l2_fsze_G:.4f},{l1_sze_G:.4f},{min_l2_usze:.4f},{l2_fmusze_GT:.4f},{l1_fmusze_GT:.4f},{kvs_fmusze:.4f}\n"
            print(row)
            if write_csv:
                with open(CSV_PATH, 'a') as f:
                    f.write(row)


elapsed = time.time() - tm
print(f"[i] Time elapsed: {elapsed}")

ipdb.set_trace()

