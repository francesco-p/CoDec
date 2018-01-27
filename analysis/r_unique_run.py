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
        #if k > max_k:
            max_k = k
            max_idx = keci[k][2]

    return max_k


######################################################
################ Main script code ####################
######################################################

#dset = "./data_unique_run/npz/cahepth.npz"
#dset = "./data_unique_run/npz/cahepph.npz"
#dset = "./data_unique_run/npz/cagrqc.npz"
#dset = "./data_unique_run/npz/wiki_vote.npz"
#dset = "./data_unique_run/npz/corecipient.npz"
#dset = "./data_unique_run/npz/p2p-gnutella.npz"

dset = "./data_unique_run/npz/openflights.npz"

#dset = "./data_unique_run/npz/movielens.npz"
#dset = "./data_unique_run/npz/reactome.npz"
#dset = "./data_unique_run/npz/email-Eu-core.npz"
#dset = "./data_unique_run/npz/facebook.npz"

dset_name = dset.split('/')[-1][:-4]
data = np.load(dset)

pu.cprint(dset)

G = data['G']

repetitions = 1 

refinement = 'indeg_guided'

print(f"n={G.shape[0]}")
print("[+] Filtering G")
fG = ndimage.median_filter(G,23)

density = pd.density(G, weighted=True)
print(f"[+] Density of G {density}")


header = f"n,density,n_partitions,k,nirr,epsilon,sze_idx,l2_sze_G,l2_fsze_fG,l2_sze_fG,l2_fsze_G,min_l2_usze_G,desired_th,tpartition,treconstructed,tfiltered,tremove\n"
#if not os.path.isfile(f"./data_unique_run/csv/{dset_name}.csv"):
    #with open(f"./data_unique_run/csv/{dset_name}.csv", 'w') as f:
        #f.write(header)

for repetition in range(repetitions):
    print(f"{repetition}/{repetitions}")

    data = {}
    data['G'] = G
    data['GT'] = []
    data['bounds'] = [0.05, 0.5]
    data['labels'] = []
    s = SensitivityAnalysis(data, refinement)
    s.verbose = True
    s.is_weighted = True

    tm = time.time()

    print("[+] Finding bounds ...")
    bounds = s.find_bounds()
    print(f"[i] Bounds : {bounds}")

    print(f"[+] Finding partitions {refinement}...")
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


    print("[+] Reconstruction t:0")
    sze = s.reconstruct_mat(0, classes, k, reg_list)


    treconstructed = time.time() - tm
    print(f"[T] Reconstructed: {treconstructed}")


    print("[+] Filtering")
    fsze = ndimage.median_filter(sze,23)


    tfiltered = time.time() - tm
    print(f"[T] sze filtered: {tfiltered}")

    l2_sze_G = s.L2_metric(sze)
    l2_fsze_fG = np.linalg.norm(fG-fsze)/fG.shape[0]
    l2_sze_fG = np.linalg.norm(fG-sze)/fG.shape[0]
    l2_fsze_G = np.linalg.norm(G-fsze)/G.shape[0]

    pu.cprint(f"l2 sze G:{l2_sze_G}\nl2 fsze fG:{l2_fsze_fG}\nl2 fsze G:{l2_fsze_G}\nl2 sze fG:{l2_sze_fG}")

    pu.plot_graphs([G, sze, fsze, fG ], [f"{dset_name} G", f"sze {k}", "filtered sze", "filtered G"])#, FILENAME=f"./data_unique_run/plots/{dset_name}.eps", save=True)


    print(f"[+] Remove weights with thresholds")
    min_l2_usze_G = 100
    desired_th = -1
    for i in np.arange(0,1,0.05):
        usze = (sze>(i*np.max(sze)))
        l2_usze_G = np.linalg.norm(G-usze)/G.shape[0]

        if l2_usze_G < min_l2_usze_G:
            desired_th = i
            min_l2_usze_G = l2_usze_G

        density_usze = pd.density(usze, weighted=False)
        print("{:.2f} : {:.4f} : {:.4f} : {:.4f}".format(i, density_usze, density, l2_usze_G))

    tremove = time.time() - tm
    print(f"[T] Unweight: {tremove}")

    row = f"{G.shape[0]},{density:.4f},{n_partitions},{k},{nirr},{epsilon:.5f},{sze_idx:.4f},{l2_sze_G:.4f},{l2_fsze_fG:.4f},{l2_sze_fG:.4f},{l2_fsze_G:.4f},{min_l2_usze_G:.4f},{desired_th:.2f},{tpartition:.2f},{treconstructed:.2f},{tfiltered:.2f},{tremove:.2f}\n"

    #with open(f"./data_unique_run/csv/{dset_name}.csv", 'a') as f:
        #f.write(row)

ipdb.set_trace()
