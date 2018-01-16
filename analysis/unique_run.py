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

def replicator(A, x, inds, tol, max_iter):
    error = tol + 1.0
    count = 0
    while error > tol and count < max_iter:
        x_old = np.copy(x)
        for i in inds:
            x[i] = x_old[i] * (A[i] @ x_old)
        x /= np.sum(x)
        error = np.linalg.norm(x - x_old)
        count += 1
    return x


def dominant_sets(graph_mat, max_k=0, tol=1e-5, max_iter=1000):
    graph_cardinality = graph_mat.shape[0]
    if max_k == 0:
        max_k = graph_cardinality
    clusters = np.zeros(graph_cardinality)
    already_clustered = np.full(graph_cardinality, False, dtype=np.bool)

    for k in range(max_k):
        if graph_cardinality - already_clustered.sum() <= ceil(0.05 * graph_cardinality):
            break
        # 1000 is added to obtain more similar values when x is normalized
        # x = np.random.random_sample(graph_cardinality) + 1000.0
        x = np.full(graph_cardinality, 1.0)
        x[already_clustered] = 0.0
        x /= x.sum()

        y = replicator(graph_mat, x, np.where(~already_clustered)[0], tol, max_iter)
        cluster = np.where(y >= 1.0 / (graph_cardinality * 1.5))[0]
        already_clustered[cluster] = True
        clusters[cluster] = k
    clusters[~already_clustered] = k
    return clusters

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



tm = time.time()

repetitions = 5 

n = 100
refinement = 'indeg_guided'

intranoise = [0]
internoise = [0.1, 0.3, 0.4, 0.5, 0.7, 0.8]
#intranoise = np.arange(0,1,0.05) 

inter_v = 1 
intra_v = 0

header = f"n,graph_name,internoise,intranoise,density,k,epsilon,sze_idx,nirr,refinement,best_threshold,l2_G_GT,l1_G_GT,kvs_G,l2_fG_GT,l1_fG_GT,kvs_fG,l2_sze_GT,l1_sze_GT,kvs_sze,l2_fsze_GT,l1_fsze_GT,kvs_fsze\n"
CSV_PATH = f"./data_unique_run/csv/{n}.csv"
#with io.open(CSV_PATH, 'w') as f:
    #f.write(header)


tensor = np.empty((repetitions, 4, len(internoise), ), dtype="float")

for repetition in range(0,repetitions):

    pu.cprint(f"Repetition {repetition}")
    
    kvs_sze_arr = []
    kvs_fsze_arr = []
    kvs_G_arr = []
    kvs_fG_arr = []


    for intranoise_lvl in intranoise:

        for internoise_lvl in internoise:

            clusters = [int(n*0.45), int(n*0.2), int(n*0.2), int(n*0.15)]
            #nc = n // 4
            #clusters = [nc]*4

            print(f"[+] Gen Graph n:{n} inter:{internoise_lvl} intra:{intranoise_lvl}")
            G, GT, labels = pd.custom_cluster_matrix(n, clusters, internoise_lvl, inter_v, intranoise_lvl, intra_v)
            #G, GT, labels = pd.custom_crazy_cluster_matrix(n, clusters, internoise_lvl, inter_v, intranoise_lvl, 0)
            #G, GT, labels = pd.custom_cluster_matrix(n, [n], 1, 1, 0.1, 0)
            #labels = dominant_sets(G).astype('int64')


            density = pd.density(G, weighted=True)

            graph_name = f"{n:06d}_{internoise_lvl:.2f}_{intranoise_lvl:.2f}_{repetition:03d}"
            print(graph_name)

            data = {}
            data['G'] = G
            data['GT'] = GT
            data['bounds'] = [0.10, 0.5]
            data['labels'] = labels

            s = SensitivityAnalysis(data, refinement)
            s.verbose = True
            s.is_weighted = True
            #s.drop_edges_between_irregular_pairs = False

            print(f"[+] Density of G {density}")

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

            #print("[+] Running...")
            #regular, k, classes, sze_idx, reg_list , nirr = s.run_alg(0.31) #0.25 n20000
            #print(f"[+] Partition found: {regular}, {k}, {sze_idx}, {nirr}")

            l2_G_GT = s.L2_metric_GT(G)
            l1_G_GT = s.L1_metric_GT(G)
            kvs_G = s.KVS_metric(G)
            pu.cprint(f"kvs_G:{kvs_G}", COL=pu.RED)
            kvs_G_arr.append(kvs_G)

            #DS_G = s.DS_metric(G)
            #pu.cprint(f"DS_G:{DS_G}", COL=pu.RED)

            print("[+] Filtering G")
            fG = ndimage.median_filter(G,23)
            l2_fG_GT = s.L2_metric_GT(fG)
            l1_fG_GT = s.L1_metric_GT(fG)
            kvs_fG = s.KVS_metric(fG)
            pu.cprint(f"kvs_fG:{kvs_fG}", COL=pu.RED)
            kvs_fG_arr.append(kvs_fG)

            #fDS_G = s.DS_metric(fG)
            #pu.cprint(f"fDS_G:{fDS_G}", COL=pu.RED)


            print("[+] Reconstruction")
            measures = [-1]*4
            #for t in [0, 0.2, 0.4, 0.6, density+0.05, density+0.03]: #np.arange(0,1, 0.05):
            for t in [0]: #np.arange(0,1, 0.05): # CHEATING WE DONT KNOW GT
                print(f"  Threshold: {t:.2f}")
                sze_rec = s.reconstruct_mat(t, classes, k, reg_list)

                l2_sze_GT = s.L2_metric_GT(sze_rec)
                l1_sze_GT = s.L1_metric_GT(sze_rec)
                kvs_sze = s.KVS_metric(sze_rec)
                pu.cprint(f"kvs_sze:{kvs_sze}", COL=pu.GRN)
                kvs_sze_arr.append(kvs_sze)

                #DS_sze = s.DS_metric(sze_rec)
                #pu.cprint(f"DS_sze:{DS_sze}", COL=pu.GRN)

                print("[+] Filtering sze")
                fsze = ndimage.median_filter(sze_rec,23)
                l2_fsze_GT = s.L2_metric_GT(fsze)
                l1_fsze_GT = s.L1_metric_GT(fsze)
                kvs_fsze = s.KVS_metric(fsze)
                pu.cprint(f"kvs_fsze:{kvs_fsze}", COL=pu.GRN)
                kvs_fsze_arr.append(kvs_fsze)

                #DS_fsze = s.DS_metric(fsze)
                #pu.cprint(f"DS_fsze:{DS_fsze}", COL=pu.GRN)

                print(f"ari: {kvs_fG:.4f}-{kvs_fsze:.4f} | l2:{l2_fG_GT:.4f}-{l2_fsze_GT:.4f} | l1:{l1_fG_GT:.4f}-{l1_fsze_GT:.4f}")

                if kvs_fsze > measures[0]:
                    measures = [kvs_fsze, l2_fsze_GT, l1_fsze_GT, t]
                    #np.savez(f"./data_unique_run/{graph_name}", fsze=fsze, sze_rec=sze_rec, GT=GT, G=G, epsilon=np.array(epsilon))

                #pu.plot_graphs([G, sze_rec, fsze, fG ], ["G", f"sze_rec {k}", "filtered sze", "filtered G"])

            kvs_fsze = measures[0]
            l2_fsze_GT = measures[1]
            l1_fsze_GT = measures[2]
            best_threshold = measures[3]

            row = f"{n},{graph_name},{internoise_lvl:.2f},{intranoise_lvl:.2f},{density:.4f},{k},{epsilon:.10f},{sze_idx:.5f},{nirr},{refinement},{best_threshold:.2f},{l2_G_GT:.4f},{l1_G_GT:.4f},{kvs_G:.4f},{l2_fG_GT:.4f},{l1_fG_GT:.4f},{kvs_fG:.4f},{l2_sze_GT:.4f},{l1_sze_GT:.4f},{kvs_sze:.4f},{l2_fsze_GT:.4f},{l1_fsze_GT:.4f},{kvs_fsze:.4f}\n"

            print(row, end="")
            with io.open(CSV_PATH, 'a') as f:
                f.write(row)


    #ipdb.set_trace()

    tensor[repetition] = np.array([kvs_sze_arr, kvs_fsze_arr, kvs_G_arr, kvs_fG_arr])



elapsed = time.time() - tm
print(f"[i] Time elapsed: {elapsed}")
ipdb.set_trace()

tensor_sum = tensor.sum(0) / repetitions


#[np.array(kvs_sze_arr), np.array(kvs_fsze_arr), [np.array(kvs_G_arr).mean()] * repetitions


plt.plot(internoise, tensor_sum[0], label=f"ARI(rec)")
plt.plot(internoise, tensor_sum[1], label="ARI(rec+filter)")
plt.plot(internoise, tensor_sum[2], label="ARI(G)")
plt.plot(internoise, tensor_sum[3], label="ARI(G+filter)")

plt.title(f"n={n}, intranoise 0%, inbalanced clusters")
plt.ylabel('ARI')
plt.xlabel('Internoise')
plt.legend()

plt.savefig("./data_unique_run/plots/{n}_internoise.eps")

plt.show()

