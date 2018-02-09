"""
File: synt.py
Coding: UTF-8
Author: lakj
Indentation : 4spaces
"""
from sensitivity_analysis import SensitivityAnalysis
import numpy as np
import matplotlib.pyplot as plt
import process_datasets as pd
import putils as pu
import scipy.io as spio
from scipy import ndimage
import time
import ipdb
import os.path


def best_partition(partitions):
    """ Selects the best partition (highest sze_idx)
    :param partitions: dictionary
    :returns: the k of the best partition
    """
    max_idx = -1
    max_k = -1

    for k in partitions.keys():
        if partitions[k][2] > max_idx:
        #if k > max_k:
            max_k = k
            max_idx = partitions[k][2]

    return max_k


################ Main script code ####################

# I dataset 

dset = "./data_unique_run/npz/mats/NISTGEKforMarco.mat"


mat = spio.loadmat(dset)
print(mat)

# Devi sapere la key giusta per accedere alla matrice
# lo vedi da matlab oppure fai print(mat)
G = mat['GEK']


# Generare il GT con questa utility che genera grafi G e ritorna anche GT e labels
aux, GT, labels = pd.custom_cluster_matrix(5000, [500]*10, 0,0,0,0)

dset_name = dset.split('/')[-1][:-4]
print(f"[+] {dset} n={G.shape[0]} loaded, GT created.")

print("[+] Filtering G --> fG, it takes a while...")
fG = ndimage.median_filter(G,23)

density = pd.density(G, weighted=True)
print(f"[+] Density of G:{density}")

repetitions = 1
refinement = 'indeg_guided'

for repetition in range(repetitions):
    print(f"[+] r={repetition+1}/{repetitions}")

    data = {}

    data['G'] = G
    data['GT'] = GT # you can provide a empty [] if you don't have it
    data['labels'] = labels #you can provide [] if you don't have it

    # Epsilon range
    data['bounds'] = [0.05, 0.5]
    s = SensitivityAnalysis(data, refinement)
    s.verbose = True
    s.is_weighted = True
    # If fast search then the search stops after finding the first false partition
    s.fast_search = True
    # Number of epsilon selected in the range
    s.tries = 20  
    #s.drop_edges_between_irregular_pairs = False

    tm = time.time()

    # If you specify the epsilon range then it does not
    # find the bounds, but the find_bounds() still need to be called
    print("[+] Finding bounds ...")
    bounds = s.find_bounds()
    print(f"[i] Bounds : {bounds}")

    print(f"[+] Finding partitions")
    partitions = s.find_partitions()

    tpartition = time.time() - tm
    print(f"[T] Partitions found: {tpartition}")

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


    print("[+] Reconstruction --> SZE")
    sze = s.reconstruct_mat(0, classes, k, reg_list)

    treconstructed = time.time() - tm
    print(f"[T] Reconstructed: {treconstructed}")

    print("[+] Filtering SZE --> FSZE")
    fsze = ndimage.median_filter(sze,23)

    tfiltered = time.time() - tm
    print(f"[T] sze filtered: {tfiltered}")

    # Metrics
    l2_sze_G = s.L2_metric(sze)
    l2_fsze_G = s.L2_metric(fsze)

    l2_fsze_fG = np.linalg.norm(fG-fsze)/fG.shape[0]
    l2_sze_fG = np.linalg.norm(fG-sze)/fG.shape[0]

    # Requires GT if you don't have the GT then
    # just comment these lines
    l2_sze_GT = s.L2_metric_GT(sze)
    l2_fsze_GT = s.L2_metric_GT(fsze)
    l2_G_GT = s.L2_metric_GT(G)
    KVS_G = s.KVS_metric(G)
    KVS_sze = s.KVS_metric(sze)
    KVS_fsze = s.KVS_metric(fsze)


    print(f"[+] Unweighting SZE matrix by minimizing L2")
    min_l2_usze = 100
    desired_th = -1

    for i in np.arange(0,1,0.05):
        usze = (sze>(i*np.max(sze)))

        # Unweighting by minimizing L2(sze,G) OR L2(sze,GT)
        #l2_usze = np.linalg.norm(G-usze)/G.shape[0]
        l2_usze = np.linalg.norm(GT-usze)/GT.shape[0]

        if l2_usze < min_l2_usze:
            desired_th = i
            min_l2_usze = l2_usze
            musze = usze

        density_usze = pd.density(usze, weighted=False)

    tunweight = time.time() - tm
    print(f"[T] Unweight: {tunweight}")

    print(f"[+] Best thrreshold:{desired_th:.2f} minL2:{min_l2_usze:.4f}")

    print("[+] Filtering unweighted SZE --> fUSZE")
    fmusze = ndimage.median_filter(musze,23)

    print("[P] plots the matrices")
    pu.plot_graphs([G, sze, fsze, fG, musze, fmusze ], [f"{dset_name} G", f"sze {k}", "filtered sze", "filtered G", "musze", "fmusze"])

    l2_fmusze_G = s.L2_metric(fmusze)
    l2_musze_G = s.L2_metric(musze)

    # Require GT
    l2_fmusze_GT = s.L2_metric_GT(fmusze)
    l2_musze_GT = s.L2_metric_GT(musze)
    KVS_fmusze = s.KVS_metric(fmusze)
    KVS_musze = s.KVS_metric(musze)

    # Print of metrics some requires GT :(
    print(f"l2_sze_G:{l2_sze_G:.4f}\n l2_fsze_G:{l2_fsze_G:.4f}\n l2_fsze_fG:{l2_fsze_fG:.4f}\n l2_sze_fG:{l2_sze_fG:.4f}\n l2_sze_GT:{l2_sze_GT:.4f}\n l2_fsze_GT:{l2_fsze_GT:.4f}\n l2_G_GT:{l2_G_GT:.4f}\n KVS_sze:{KVS_sze:.4f}\n KVS_fsze:{KVS_fsze:.4f}\n KVS_G:{KVS_G:.4f}\n KVS_fmusze:{KVS_fmusze:.4f}\n KVS_musze:{KVS_musze:.4f}\n l2_fmusze_GT:{l2_fmusze_GT:.4f}\n l2_fmusze_G:{l2_fmusze_G:.4f}\n l2_musze_GT:{l2_musze_GT:.4f}\n l2_musze_G:{l2_musze_G:.4f}\n ")


# Generate the reduced matrix
red = s.generate_reduced_sim_mat(k,epsilon,classes,reg_list)

# Save data in /tmp folder
spio.savemat("/tmp/data.mat", {'sze':sze, 'k':k, 'fsze':fsze,'red':red}, do_compression=True)


