from codec import Codec
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import process_datasets as pd
import putils as pu
import time
import os.path
import metrics
from stats import Stats

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
dset = "./data_unique_run/npz/facebook.npz"
#dset = "./data_unique_run/npz/email-Enron.npz"

# Dataset spagnoli
#dset = "./data_unique_run/npz/Enist.npz"
#dset = "./data_unique_run/npz/k25p4EK.npz"
#dset = "./data_unique_run/npz/we2000_1.npz"
#dset = "./data_unique_run/npz/we2000_2.npz"
#dset = "./data_unique_run/npz/RRWCoil75.npz"


print(dset)
dset_name = dset.split('/')[-1][:-4]
data = np.load(dset)

GT = []
labels = []

try:
    GT = data['GT']
    labeling = data['labels']
except:
    print("[x] No GT found, No labeling found")

G = data['G']

n = G.shape[0]
repetitions = 1
refinement = 'indeg_guided'
ksize = 23

s = Stats(f"/tmp/{dset_name}.csv")
c = Codec(0.1, 0.4, 40)

for r in range(repetitions):
    print(f"### r={r} ###")

    tm = time.time()
    k, epsilon, classes, sze_idx, reg_list, nirr = c.compress(G, refinement)
    tcompression = time.time() - tm

    sze = c.decompress(G, 0, classes, k, reg_list)
    tdecompression = time.time() - tm

    fsze = c.post_decompression(sze, ksize)
    tpostdecompression = time.time() - tm

    red = c.reduced_matrix(G, k, epsilon, classes, reg_list)

    telapsed = [tcompression, tdecompression, tpostdecompression]

    sd_red_G = metrics.spectral_dist(G, red)
    print(f"SD RED G {sd_red_G:.4f}")

    #if len(GT) != 0:
        #s.real_stats_GT(k, epsilon, sze_idx, nirr, refinement, G, GT, labeling, sze, fsze, red, telapsed, write=False, plot=True, pp=True)

    #jelse:
        #s.real_stats(k, epsilon, sze_idx, nirr, refinement, G, sze, fsze, red, telapsed, write=False, plot=True, pp=True)

    pu.plot_graphs([G, sze, fsze],["G", "sze", "fsze"])

