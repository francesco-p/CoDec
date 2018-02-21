import metrics
import putils
import process_datasets as pd
import numpy as np
import os
import putils as pu


class Stats:
    def __init__(self, filename):
        self.filename = filename
        self.header = False

    def unweight(self, fsze, graph):
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


    def real_header(self):
        header = "n,density,k,epsilon,sze_idx,nirr,refinement,tcompression,tdecompression,tpostdecompression,l2_sze_G,l2_fsze_G,l1_sze_G,l1_fsze_G,l2_usze_G,th_usze_G,l2_ufsze_G,th_ufsze_G\n"
        print(f"[Stats] .csv Filename: {self.filename}")
        if not os.path.isfile(self.filename):
            with open(self.filename, 'w') as f:
                f.write(header)


    def real_stats(self, k, epsilon, sze_idx, nirr, refinement, G, sze, fsze, telapsed, write=False, plot=True, pp=True):

        print("###### Statistics ######")
        if not self.header:
            self.real_header()

        tcompression = telapsed[0]
        tdecompression = telapsed[1] - telapsed[0]
        tpostdecompression = telapsed[2] - telapsed[1]
        n = G.shape[0]
        density = pd.density(G)

        l2_sze_G = metrics.l2(sze, G)
        l2_fsze_G = metrics.l2(fsze, G)

        l1_sze_G = metrics.l1(sze, G)
        l1_fsze_G = metrics.l1(fsze, G)

        ufsze_G, l2_ufsze_G, th_ufsze_G = self.unweight(fsze, G)
        usze_G, l2_usze_G, th_usze_G = self.unweight(sze, G)

        if pp:
            print(f"G density: {density}")
            print(f"l2 FSZE G: {l2_fsze_G:.4f}")
            print(f"l1 FSZE G: {l1_fsze_G:.4f}")
            print(f"l2 UFSZE G: {l2_ufsze_G:.4f}")
            print(f"[T] Time Compression: {tcompression}")
            print(f"[T] Time Decompression: {tdecompression}")
            print(f"[T] Time Post-Decompression Filtering: {tpostdecompression}")

        if write:
            row = f"{n},{density:.2f},{k},{epsilon:.6f},{sze_idx:.4f},{nirr},{refinement},{tcompression:.2f},{tdecompression:.2f},{tpostdecompression:.2f},{l2_sze_G:.4f},{l2_fsze_G:.4f},{l1_sze_G:.4f},{l1_fsze_G:.4f},{l2_usze_G:.4f},{th_usze_G:.2f},{l2_ufsze_G:.4f},{th_ufsze_G:.2f}\n"

            with open(self.filename, 'a') as f:
                f.write(row)

        if plot:
            pu.plot_graphs([G, sze, fsze, ufsze_G], [f"G", f"SZE k:{k}", "FSZE", "UFSZE-G"])



    def synth_header(self):
        header = "n,imbalanced,num_c,internoiselvl,intranoiselvl,density,k,epsilon,sze_idx,nirr,refinement,tcompression,tdecompression,tpostdecompression,kvs_sze,kvs_fsze,l2_sze_G,l2_fsze_G,l1_sze_G,l1_fsze_G,l2_sze_GT,l2_fsze_GT,l1_sze_GT,l1_fsze_GT,l2_usze_GT,th_usze_GT,l2_ufsze_GT, th_ufsze_GT,l2_usze_G, th_usze_G,l2_ufsze_G, th_ufsze_G\n"
        print(f"[Stats] .csv Filename: {self.filename}")
        if not os.path.isfile(self.filename):
            with open(self.filename, 'w') as f:
                f.write(header)



    def synth_stats(self, imbalanced, num_c, internoiselvl, intranoiselvl, k, epsilon, sze_idx, nirr, refinement, G, GT, labeling, sze, fsze, telapsed, write=False, plot=True, pp=True):

        print("###### Statistics ######")

        if not self.header:
            self.synth_header()

        tcompression = telapsed[0]
        tdecompression = telapsed[1] - telapsed[0]
        tpostdecompression = telapsed[2] - telapsed[1]
        n = G.shape[0]
        density = pd.density(G)

        l2_fsze_GT = metrics.l2(fsze, GT)
        l2_fsze_G = metrics.l2(fsze, G)
        l2_sze_G = metrics.l2(sze, G)
        l2_sze_GT = metrics.l2(sze, GT)

        l1_fsze_GT = metrics.l1(fsze, GT)
        l1_fsze_G = metrics.l1(fsze, G)
        l1_sze_G = metrics.l1(sze, G)
        l1_sze_GT = metrics.l1(sze, GT)

        kvs_fsze = metrics.ARI_KVS(fsze, labeling)
        kvs_sze = metrics.ARI_KVS(sze, labeling)

        ufsze_G, l2_ufsze_G, th_ufsze_G = self.unweight(fsze, G)
        ufsze_GT, l2_ufsze_GT, th_ufsze_GT = self.unweight(fsze, GT)

        usze_G, l2_usze_G, th_usze_G = self.unweight(sze, G)
        usze_GT, l2_usze_GT, th_usze_GT = self.unweight(sze, GT)

        if pp:
            print(f"G density: {density}")
            print(f"ARI KVS FSZE: {kvs_fsze:.4f}")
            print(f"ARI KVS SZE: {kvs_sze:.4f}")
            print(f"l2 FSZE G: {l2_fsze_G:.4f}")
            print(f"l1 FSZE G: {l1_fsze_G:.4f}")
            print(f"l2 UFSZE GT: {l2_ufsze_GT:.4f}")
            print(f"[T] Time Compression: {tcompression}")
            print(f"[T] Time Decompression: {tdecompression}")
            print(f"[T] Time Post-Decompression Filtering: {tpostdecompression}")

        if write:
            row = f"{n},{imbalanced},{num_c},{internoiselvl:.2f},{intranoiselvl:.2f},{density:.4f},{k},{epsilon:.6f},{sze_idx:.4f},{nirr},{refinement},{tcompression:.2f},{tdecompression:.2f},{tpostdecompression:.2f},{kvs_sze:.4f},{kvs_fsze:.4f},{l2_sze_G:.4f},{l2_fsze_G:.4f},{l1_sze_G:.4f},{l1_fsze_G:.4f},{l2_sze_GT:.4f},{l2_fsze_GT:.4f}, {l1_sze_GT:.4f},{l1_fsze_GT:.4f},{l2_usze_GT:.4f},{th_usze_GT:.2f},{l2_ufsze_GT:.4f},{th_ufsze_GT:.2f},{l2_usze_G:.4f},{th_usze_G:.2f},{l2_ufsze_G:.4f},{th_ufsze_G:.2f}\n"
            with open(self.filename, 'a') as f:
                f.write(row)

        if plot:
            pu.plot_graphs([G, sze, fsze, ufsze_GT], [f"G", f"SZE k:{k}", "FSZE", "UFSZE-GT"])

