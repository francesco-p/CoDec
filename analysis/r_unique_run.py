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
dset = "./data_unique_run/npz/openflights.npz"
#dset = "./data_unique_run/npz/movielens.npz"
#dset = "./data_unique_run/npz/reactome.npz"
#dset = "./data_unique_run/npz/email-Eu-core.npz"
#dset = "./data_unique_run/npz/facebook.npz"
#dset = "./data_unique_run/npz/email-Enron.npz"

# Dataset spagnoli
#dset = "./data_unique_run/npz/Enist.npz"
#dset = "./data_unique_run/npz/k25p4EK.npz"


dset_name = dset.split('/')[-1][:-4]
data = np.load(dset)

pu.cprint(dset)

G = data['G']
#GT = data['GT']
GT = [] 
#labels = data['labels']
labels = []

repetitions = 1
tries = 40
write_csv = False

refinement = 'indeg_guided'

n = G.shape[0]
print(f"n={n}")
print("[+] Filtering G")

density = pd.density(G, weighted=True)
print(f"[+] Density of G {density}")

header = f"n,density,k,epsilon,sze_idx,nirr,refinement,tpartition,treconstruction,tfiltering,l2_sze_G,l2_fsze_G,l1_sze_G,l1_fsze_G,l2_usze_G,th_usze_G,l2_ufsze_G,th_ufsze_G\n"

CSV_PATH = f"./data_unique_run/csv/realdsets/{dset_name}.csv"
print(CSV_PATH)
if not os.path.isfile(CSV_PATH) and write_csv:
    with open(CSV_PATH, 'w') as f:
        f.write(header)

for repetition in range(repetitions):
    print(f"{repetition+1}/{repetitions}")

    data = {}
    data['G'] = G
    data['GT'] = GT
    data['bounds'] = [0.08, 0.25]
    data['labels'] = labels
    s = SensitivityAnalysis(data, refinement)
    s.verbose = True
    s.is_weighted = True
    s.fast_search = True
    s.tries = tries

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

    treconstruction = time.time() - tm
    print(f"[T] Reconstructed: {treconstruction}")

    print("[+] Filtering")
    fsze = ndimage.median_filter(sze,23)

    tfiltering = time.time() - tm
    print(f"[T] sze filtered: {tfiltering}")

    l2_sze_G = s.L2_metric(sze)
    l2_fsze_G = s.L2_metric(fsze)
    l1_sze_G = s.L1_metric(sze)
    l1_fsze_G = s.L1_metric(fsze)

    usze_G, l2_usze_G, th_usze_G = best_threshold(sze, G)
    ufsze_G, l2_ufsze_G, th_ufsze_G = best_threshold(fsze, G)

    pu.plot_graphs([G, sze, fsze, usze_G, ufsze_G], [f"{dset_name} G", f"sze {k}", "fsze",  "usze-G", "ufsze-G"])#, FILENAME=f"./data_unique_run/plots/{dset_name}.eps", save=True)

    tremove = time.time() - tm
    print(f"[T] Unweight: {tremove}")


    row = f"{n},{density:.4f},{k},{epsilon:.6f},{sze_idx:.4f},{nirr},{refinement},{tpartition:.2f},{treconstruction:.2f},{tfiltering:.2f},{l2_sze_G:.4f},{l2_fsze_G:.4f},{l1_sze_G:.4f},{l1_fsze_G:.4f},{l2_usze_G:.4f},{th_usze_G:.2f},{l2_ufsze_G:.4f},{th_ufsze_G:.2f}\n"
    print(row)


    if write_csv:
        with open(CSV_PATH, 'a') as f:
            f.write(row)


# Generate the reduced matrix
#red = s.generate_reduced_sim_mat(k,epsilon,classes,reg_list)

# Save data in /tmp folder
#np.savez_compressed("/tmp/data.npz", sze=sze, fsze=fsze, red=red)
#spio.savemat("/tmp/data.mat", {'sze':sze, 'k':k, 'fsze':fsze,'red':red}, do_compression=True)

