from codec import Codec
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import process_datasets as pd
import putils as pu
import time
import os.path
import metrics
import scipy.io as spio
from stats import Stats


def unweight(fsze, graph):
    """ Unweighting of fsze matrix by minimizing l2 with respect to graph
    :param fsze: np.array((n, n)) matrix to be unweighted
    :paragm graph: np.array((n, n)) target matrix
    :return: np.array((n, n)), float, float unweighted matrix, l2 distance, value of the threshold
    """
    print(f"[+] Unweighting...")
    min_l2 = 100
    threshold = -1
    for t in np.arange(0,1,0.01):
        umat = (fsze>(t*np.max(fsze)))

        l2 = np.linalg.norm(graph-umat)/graph.shape[0]

        if l2 < min_l2:
            threshold = t
            min_l2 = l2
            bestmat = umat

    return bestmat, min_l2, threshold


"""
def unweight2(fsze, graph):
    print(f"[+] Unweighting...")
    min_l2 = 100
    threshold = -1
    for t in np.arange(0,1,0.01):
        umat = (fsze>(t*np.max(fsze)))

        l2 = metrics.spectral_dist(G, red)

        if l2 < min_l2:
            threshold = t
            min_l2 = l2
            bestmat = umat

    return bestmat, min_l2, threshold

"""



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
#dset = "./data_unique_run/npz/k25p4EK.npz"
#dset = "./data_unique_run/npz/we2000_1.npz"
#dset = "./data_unique_run/npz/we2000_2.npz"
#dset = "./data_unique_run/npz/we2000_3.npz"
#dset = "./data_unique_run/npz/knn15_2000.npz"
#dset = "./query/c20.npz"
#dset = "./data_unique_run/npz/RRWCoil75.npz"

#dsets = [ "./data_unique_run/npz/WeKnn15Logo.npz", "./data_unique_run/npz/WeKnn25Logo.npz",  "./data_unique_run/npz/WeKnn35Logo.npz", "./data_unique_run/npz/WeKnn45Logo.npz"]

#dsets = [ "./data_unique_run/npz/WeKnn25Yale.npz","./data_unique_run/npz/WeKnn35Yale.npz", "./data_unique_run/npz/WeKnn45Yale.npz"]

#dsets = [ "./data_unique_run/npz/RRWCoil75.npz"]

dsets = [ "./data_unique_run/npz/We2000.npz"]

for dset in dsets:

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


    refinement = 'indeg_guided'
    ksize = 23

    s = Stats(f"/tmp/{dset_name}.csv")

    c = Codec(0, 0.5, 20)
    c.verbose = False

    #f = []
    #g = []
    #thresholds = np.arange(0,0.7, 0.05)

    thresholds = np.arange(0,0.5,0.05)
    for t in thresholds:
        print(f"### t={t} ###")
        #G = (data['G']*10**4)>(t*np.max(data['G']))
        G = data['G'] > 0

        n = G.shape[0]

        k, epsilon, classes, sze_idx, reg_list, nirr = c.compress(G, refinement)
        sze = c.decompress(G, 0, classes, k, reg_list)
        fsze = c.post_decompression(sze, ksize)
        red = c.reduced_matrix(G, k, epsilon, classes, reg_list)


        ufsze_G, l2_ufsze_G, th_ufsze_G = unweight(fsze.astype('float32'), G.astype('float32'))
        #ufsze_G, l2_ufsze_G, th_ufsze_G = unweight2(fsze.astype('float32'), G.astype('float32'))
        print(fsze.shape)
        #kvs_fsze = metrics.ARI_KVS(fsze, labeling)
        #kvs_ufsze = metrics.ARI_KVS(ufsze_G, labeling)
        #print(f"{dset_name} ari fsze {kvs_fsze:.4f}")
        #print(f"{dset_name} ari ufsze {kvs_ufsze:.4f}")

        #pu.plot_graphs([G, sze, fsze, ufsze_G], [f"G", f"SZE k:{k}", "FSZE", "UFSZE-G"])

        #s.exp_stats(t, k, epsilon, sze_idx, nirr, refinement, G, GT, labeling, sze, fsze, red, write=False, plot=False, pp=True)
        break

    dt = {}
    dt['fsze'] = fsze
    dt['ufsze'] = ufsze_G
    dt['red'] = red
    spio.savemat(f"/tmp/{dset_name}_FT", dt, do_compression=False)

#ipdb.set_trace()

"""
g.append(metrics.ARI_KVS(G, labeling))
f.append(metrics.ARI_KVS(fsze, labeling))
sd.append(metrics.spectral_dist(G, red))
l2.append(metrics.l2(G, fsze))

#plt.plot(thresholds, l2,  label="ARI G")
plt.plot(thresholds, sd)
plt.title(dset_name)
plt.ylabel("Spectral Distance")
plt.xlabel("Threshold")
plt.legend()
plt.grid()
plt.show()

plt.plot(thresholds, l2)
plt.title(dset_name)
plt.ylabel("l2")
plt.xlabel("Threshold")
plt.legend()
plt.grid()
plt.show()



dt = {}
dt['fsze'] = fsze
dt['ufsze'] = ufsze_G
dt['red'] = red
spio.savemat(f"/tmp/{dset_name}", dt, do_compression=False)

"""
