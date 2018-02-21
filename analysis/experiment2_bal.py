import process_datasets as pd
import putils as pu
from codec import Codec
from stats import Stats
import time

n = 1000
repetitions = 5
inter_s = [0, 0.05, 0.1, 0.2, 0.3]
intra_s =  inter_s
num_cs = [2,4,8,10,16,20,40]
ksize = 23
imbalanced = False

inter_v = 1
intra_v = 0

refinement = 'indeg_guided'

cdc = Codec(0, 0.5, 20)
cdc.fast_search = False

s = Stats("/tmp/test.csv")

for repetition in range(repetitions):
    for inter in inter_s:
        for intra in intra_s:
            for num_c in num_cs:
                print(pu.to_header(f"r:{repetition+1}/{repetitions} n:{n} num_c:{num_c} inter:{inter} intra:{intra}"))
                tm = time.time()
                nc = n // num_c
                clusters = [nc]*num_c
                G, GT, labeling = pd.custom_cluster_matrix(n, clusters, inter, inter_v, intra, intra_v)
                k, epsilon, classes, sze_idx, reg_list, nirr = cdc.compress(G, refinement)
                tcompression = time.time() - tm

                sze = cdc.decompress(G, 0, classes, k, reg_list)
                tdecompression = time.time() - tm

                fsze = cdc.post_decompression(sze, ksize)
                tpostdecompression = time.time() - tm

                #red = c.reduced_matrix(k, epsilon, classes, reg_list)

                telapsed = [tcompression, tdecompression, tpostdecompression]

                s.compute_stats(imbalanced, num_c, inter, intra, k, epsilon, sze_idx, nirr, refinement, G, GT, labeling, sze, fsze, telapsed, write=True, plot=True, pp=True)


