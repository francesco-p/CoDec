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

#dset = "./data_unique_run/npz/cahepth.npz"
#dset = "./data_unique_run/npz/cahepph.npz"
#dset = "./data_unique_run/npz/cagrqc.npz"
#dset = "./data_unique_run/npz/wiki_vote.npz"
#dset = "./data_unique_run/npz/corecipient.npz"
#dset = "./data_unique_run/npz/p2p-gnutella.npz"
#dset = "./data_unique_run/npz/openflights.npz"
#dset = "./data_unique_run/npz/movielens.npz"
#dset = "./data_unique_run/npz/reactome.npz"
#dset = "./data_unique_run/npz/email-Eu-core.npz"
#dset = "./data_unique_run/npz/facebook.npz"
#dset = "./data_unique_run/npz/email-Enron.npz"

# Dataset spagnoli
#dset = "./data_unique_run/npz/Enist.npz"
dset = "./data_unique_run/npz/k25p4EK.npz"


dset_name = dset.split('/')[-1][:-4]
data = np.load(dset)

pu.cprint(dset)

G = data['G']
GT = data['GT']
labels = data['labels']

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
    data['GT'] = GT
    data['bounds'] = [0.05, 0.5]
    data['labels'] = labels
    s = SensitivityAnalysis(data, refinement)
    s.verbose = True
    s.is_weighted = True
    s.fast_search = True
    s.tries = 20

    #s.drop_edges_between_irregular_pairs = False

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
    l2_fsze_G = s.L2_metric(fsze)

    l2_fsze_fG = np.linalg.norm(fG-fsze)/fG.shape[0]
    l2_sze_fG = np.linalg.norm(fG-sze)/fG.shape[0]

    l2_sze_GT = s.L2_metric_GT(sze)
    l2_fsze_GT = s.L2_metric_GT(fsze)
    l2_G_GT = s.L2_metric_GT(G)

    print(G.dtype)
    print(sze.dtype)
    print(fsze.dtype)

    KVS_G = s.KVS_metric(G)
    KVS_sze = s.KVS_metric(sze)
    KVS_fsze = s.KVS_metric(fsze)




    print(f"[+] Remove weights with thresholds")
    min_l2_usze_G = 100
    desired_th = -1
    
    for i in np.arange(0,1,0.05):
        usze = (sze>(i*np.max(sze)))
        l2_usze_G = np.linalg.norm(G-usze)/G.shape[0]

        if l2_usze_G < min_l2_usze_G:
            desired_th = i
            min_l2_usze_G = l2_usze_G
            musze = usze

        density_usze = pd.density(usze, weighted=False)
        print("{:.2f} : {:.4f} : {:.4f} : {:.4f}".format(i, density_usze, density, l2_usze_G))


    fmusze = ndimage.median_filter(musze,23)
    pu.plot_graphs([G, sze, fsze, fG, musze, fmusze ], [f"{dset_name} G", f"sze {k}", "filtered sze", "filtered G", "musze", "fmusze"])#, FILENAME=f"./data_unique_run/plots/{dset_name}.eps", save=True)


    l2_fmusze_GT = s.L2_metric_GT(fmusze)
    l2_fmusze_G = s.L2_metric(fmusze)
    l2_musze_GT = s.L2_metric_GT(musze)
    l2_musze_G = s.L2_metric(musze)

    
    KVS_fmusze = s.KVS_metric(fmusze)
    KVS_musze = s.KVS_metric(musze)

    print(f"l2_sze_G:{l2_sze_G:.4f}\n l2_fsze_G:{l2_fsze_G:.4f}\n l2_fsze_fG:{l2_fsze_fG:.4f}\n l2_sze_fG:{l2_sze_fG:.4f}\n l2_sze_GT:{l2_sze_GT:.4f}\n l2_fsze_GT:{l2_fsze_GT:.4f}\n l2_G_GT:{l2_G_GT:.4f}\n KVS_sze:{KVS_sze:.4f}\n KVS_fsze:{KVS_fsze:.4f}\n KVS_G:{KVS_G:.4f}\n KVS_fmusze:{KVS_fmusze:.4f}\n KVS_musze:{KVS_musze:.4f}\n l2_fmusze_GT:{l2_fmusze_GT:.4f}\n l2_fmusze_G:{l2_fmusze_G:.4f}\n l2_musze_GT:{l2_musze_GT:.4f}\n l2_musze_G:{l2_musze_G:.4f}\n ")

    tremove = time.time() - tm
    print(f"[T] Unweight: {tremove}")

    row = f"{G.shape[0]},{density:.4f},{n_partitions},{k},{nirr},{epsilon:.5f},{sze_idx:.4f},{l2_sze_G:.4f},{l2_fsze_fG:.4f},{l2_sze_fG:.4f},{l2_fsze_G:.4f},{min_l2_usze_G:.4f},{desired_th:.2f},{tpartition:.2f},{treconstructed:.2f},{tfiltered:.2f},{tremove:.2f}\n"

    #with open(f"./data_unique_run/csv/{dset_name}.csv", 'a') as f:
        #f.write(row)

# Generate the reduced matrix
#red = s.generate_reduced_sim_mat(k,epsilon,classes,reg_list)

# Save data in /tmp folder
#np.savez_compressed("/tmp/data.npz", sze=sze, fsze=fsze, red=red)
#spio.savemat("/tmp/data.mat", {'sze':sze, 'k':k, 'fsze':fsze,'red':red}, do_compression=True)

