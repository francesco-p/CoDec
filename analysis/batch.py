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
import sys 
sys.path.append('../')
import conf

# Params
n = conf.batch.n
n_graphs = conf.batch.n_graphs
densities = conf.batch.densities
CSV_PATH = conf.batch.CSV_PATH
DSET_PATH = conf.batch.DSET_PATH
refinement_type = conf.batch.refinement_type

with io.open(CSV_PATH, 'w') as f:
    f.write(f"n,density,dset_id,n_partitions,k,k_epsilon,L2_distance,kld_1,kld_2,sze_idx,edge,trivial,refinement\n")

for d in densities:

    for dset_id in range(1, n_graphs+1):
        ### 2. ###
        filename = f"{n}_{d:.2f}_{dset_id}.npz"

        loaded = np.load(DSET_PATH+filename)
        data = {}
        data['G'] = loaded['G']
        data['GT'] = []
        data['bounds'] = []#loaded['bounds']
        data['labels'] = []

        ### 3. ###
        s = SensitivityAnalysis(data, refinement_type)

        bounds = s.find_bounds()
        #if not s.bounds:
            #np.savez_compressed(filename, G=data['G'], bounds=bounds)

        kec = s.find_partitions()

        if kec == {}:
            print(f"[x] {filename}")

        else:

            ### 4. ###
            max_idx = -1
            max_k = -1

            for k in kec.keys():
                if kec[k][2] > max_idx:
                    max_k = k
                    max_idx = kec[k][2]

            #print(f"[+] Partition with the highest sze_idx k: {max_k} idx: {kec[max_k][2]:.4f}")

            ### 5. ###
            sze_rec = s.reconstruct_mat(d, kec[max_k][1], max_k)

            ### 6. ###
            l2_dist = s.L2_metric(sze_rec)
            kld_1, kld_2 = s.KLdivergence_metric(sze_rec)

            sze_idx = kec[max_k][2]
            n_partitions = len(kec.keys())
            e_edge = s.bounds[0]
            e_trivial = s.bounds[1]
            k_epsilon = kec[max_k][0]
            row = f"{n},{d:.2f},{dset_id},{n_partitions},{max_k},{k_epsilon:.4f},{l2_dist:.4f},{kld_1:.4f},{kld_2:.4f},{sze_idx:.4f},{e_edge:.4f},{e_trivial:.4f},{refinement_type}\n"
            print(row, end="")
            with io.open(CSV_PATH, 'a') as f:
                f.write(row)

