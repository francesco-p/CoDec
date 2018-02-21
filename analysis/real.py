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
dset = "./data_unique_run/npz/email-Eu-core.npz"
#dset = "./data_unique_run/npz/facebook.npz"
#dset = "./data_unique_run/npz/email-Enron.npz"

# Dataset spagnoli
#dset = "./data_unique_run/npz/Enist.npz"
#dset = "./data_unique_run/npz/k25p4EK.npz"


print(dset)
dset_name = dset.split('/')[-1][:-4]
data = np.load(dset)

G = data['G']
GT = []
labels = []

try:
    GT = data['GT']
    labels = data['labels']
except:
    print("[x] No GT found, No labeling found")

repetitions = 1
refinement = 'indeg_guided'
ksize = 23
n = G.shape[0]
print(f"n={n}")

s = Stats(f"/tmp/{dset_name}.csv")
c = Codec(0, 0.5, 20)

tm = time.time()
k, epsilon, classes, sze_idx, reg_list, nirr = c.compress(G, refinement)
tcompression = time.time() - tm

sze = c.decompress(G, 0, classes, k, reg_list)
tdecompression = time.time() - tm

fsze = c.post_decompression(sze, ksize)
tpostdecompression = time.time() - tm

#red = c.reduced_matrix(k, epsilon, classes, reg_list)

telapsed = [tcompression, tdecompression, tpostdecompression]

s.real_stats(k, epsilon, sze_idx, nirr, refinement, G, sze, fsze, telapsed, write=True, plot=True, pp=True)

