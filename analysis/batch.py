"""
File: batch2.py
Description:
 1. fix density d and size of the graph n
 2. process the 500 datasets
 3. find bounds and partitions
 4. for each partition then take the one with the maximum idx
 5. reconstruct the matrix with threshold 0
 6. compute the measures
 7. plot foreach density
Coding: UTF-8
Author: lakj
"""
import matplotlib.pyplot as plt
import numpy as np
from sensitivity_analysis import SensitivityAnalysis
import io
import putils as pu
from scipy import ndimage
import process_datasets as pd
import sys 
sys.path.append('../')
import conf



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


# Params
n = conf.batch.n
n_graphs = conf.batch.n_graphs
densities = conf.batch.densities
CSV_PATH = conf.batch.CSV_PATH
DSET_PATH = conf.batch.DSET_PATH
refinement_type = conf.batch.refinement_type


#header = f"n,density,dset_id,n_partitions,k,epsilon,l2,l1,kld_1,kld_2,sze_idx,min_l2_usze_G,desired_th_usze,min_l2_ufsze_G,desired_th_ufsze,refinement_type\n"
#with io.open(CSV_PATH, 'w') as f:
    #f.write(header)

for d in densities:

    n_graphs = 1 
    for dset_id in range(1, n_graphs+1):
        ### 2. ###
        filename = f"{n}_{d:.2f}_{dset_id}.npz"

        loaded = np.load(DSET_PATH+filename)
        G = loaded['G']
        data = {}
        data['G'] = G
        data['GT'] = []
        data['bounds'] = []#loaded['bounds']
        data['labels'] = []

        ### 3. ###
        s = SensitivityAnalysis(data, refinement_type)

        bounds = s.find_bounds()
        #if not s.bounds:
            #np.savez_compressed(filename, G=data['G'], bounds=bounds)

        partitions = s.find_partitions()

        if partitions == {}:
            print(f"[x] {filename}")

        else:

            ### 4. ###
            n_partitions = len(partitions.keys()) 
            print(f"[i] {n_partitions} partitions found")
            k = best_partition(partitions)

            epsilon = partitions[k][0]
            classes = partitions[k][1]
            sze_idx = partitions[k][2]
            reg_list = partitions[k][3]
            nirr = partitions[k][4]
            print(f"[+] Best partition - k:{k} epsilon:{epsilon:.4f} sze_idx:{sze_idx:.4f} irr_pairs:{nirr}")

            ### 5. ###
            sze = s.reconstruct_mat(0, classes, k, reg_list)

            ### 6. ###
            l2_dist = s.L2_metric(sze)
            l1_dist = s.L1_metric(sze)
            kld_1, kld_2 = s.KLdivergence_metric(sze)

            fsze = ndimage.median_filter(sze,23)

            min_l2_ufsze_G = 100
            desired_th_ufsze = -1
            for i in np.arange(0,1,0.005):
                ufsze = (fsze>(i*np.max(fsze)))
                l2_ufsze_G = np.linalg.norm(G-ufsze)/G.shape[0]
                pu.cprint(f"{l2_ufsze_G}", COL=pu.BLU)

                density_ufsze = pd.density(ufsze, weighted=False)

                if l2_ufsze_G < min_l2_ufsze_G:
                    desired_th_ufsze = i
                    min_l2_ufsze_G = l2_ufsze_G
                    mufsze = ufsze

            min_l2_usze_G = 100
            desired_th_usze = -1
            for i in np.arange(0,1,0.005):
                usze = (sze>(i*np.max(sze)))
                l2_usze_G = np.linalg.norm(G-usze)/G.shape[0]
                pu.cprint(f"{l2_usze_G}", COL=pu.BLU)

                density_usze = pd.density(usze, weighted=False)

                if l2_usze_G < min_l2_usze_G:
                    desired_th_usze = i
                    min_l2_usze_G = l2_usze_G
                    musze = usze

            #pu.plot_graphs([G, sze, mufsze, musze],[f"G {d}", f"sze {k}", "fusze", "musze"])


            row = f"{n},{d:.2f},{dset_id},{n_partitions},{k},{epsilon:.4f},{l2_dist:.4f},{l1_dist:.4f},{kld_1:.4f},{kld_2:.4f},{sze_idx:.4f},{min_l2_usze_G:.4f},{desired_th_usze:.2f},{min_l2_ufsze_G:.4f},{desired_th_ufsze:.2f},{refinement_type}\n"
            print(row, end="")
            with io.open(CSV_PATH, 'a') as f:
                f.write(row)

