"""
File: sensitivity_analysis.py
Description: This class performs a sensitivity analysis of the Szemeredi algorithm
Coding: UTF-8
Author: lakj
"""

import numpy as np
import matplotlib.pyplot as plt
#import matlab.engine we need to install it for the virtualenv
#import ipdb
import scipy.stats as spst
from sklearn import metrics
import sys
sys.path.insert(1, '../graph_reducer/')
import szemeredi_lemma_builder as slb
import refinement_step as rs
from math import ceil


class SensitivityAnalysis:

    def __init__(self, dset, refinement):

        # G GT
        self.set_dset(dset)

        # Find bounds parameters
        self.min_k = 4
        self.min_step = 0.01#0.0001 #0.00001
        self.tries = 20
        self.fast_search  = True # stops searching for partitions if irregular and > k

        self.epsilons = []

        # SZE algorithm parameters
        self.kind = "alon"
        self.is_weighted = False
        self.random_initialization = True

        if refinement == 'degree_based':
            self.refinement = rs.degree_based
        elif refinement == 'indeg_guided':
            self.refinement = rs.indeg_guided
        else:
            self.refinement = rs.degree_based

        self.drop_edges_between_irregular_pairs = True

        # SZE running parameters
        self.iteration_by_iteration = False
        self.sze_verbose = False
        self.compression = 0.05

        # Reconstruction parameters
        self.indensity_preservation = False

        # Matlab eng
        self.eng = None

        # Global print
        self.verbose = False


    def set_dset(self, dset):
        """ Change dataset
        :param dset: the new dictionary hoding G GT and the bounds
        """
        self.dset = dset
        #self.G = self.dset['G'] / self.dset['G'].max()
        self.G = self.dset['G']
        self.GT = self.dset['GT']
        self.labels = self.dset['labels']
        self.bounds = list(self.dset['bounds']) # to pass the test in find bounding epsilons


    def run_alg(self, epsilon):
        """ Creates and run the szemeredi algorithm with a particular dataset
        if the partition found is regular, its cardinality, and how the nodes are partitioned
        :param epsilon: float, the epsilon parameter of the algorithm
        :returns: (bool, int, np.array)
        """
        self.srla = slb.generate_szemeredi_reg_lemma_implementation(self.kind, self.G, epsilon,
                                                                    self.is_weighted, self.random_initialization,
                                                                    self.refinement, self.drop_edges_between_irregular_pairs)
        return self.srla.run(iteration_by_iteration=self.iteration_by_iteration, verbose=self.sze_verbose, compression_rate=self.compression)


    def find_trivial_epsilon(self, epsilon1, epsilon2):
        """ Performs binary search to find the best trivial epsilon candidate:
        the first epsilon for which k=2
        del statements are essential to free unused memory
        :returns: float epsilon
        """
        step = (epsilon2 - epsilon1)/2.0
        if step < self.min_step:
            return epsilon2
        else:
            epsilon_middle = epsilon1 + step
            regular, k, classes, sze_idx, reg_list , nirr = self.run_alg(epsilon_middle)
            if self.verbose:
                print(f"    |{epsilon1:.6f}-----{epsilon_middle:.6f}------{epsilon2:.6f}| {k} {regular}")

            if regular:
                if k==self.min_k:
                    del self.srla
                    return self.find_trivial_epsilon(epsilon1, epsilon_middle)
                if k>self.min_k: # could be an else
                    #self.epsilons.append(epsilon_middle)
                    del self.srla
                    return self.find_trivial_epsilon(epsilon_middle, epsilon2)
                else:
                    del self.srla
                    return -1 # WTF... just in case
            else:
                del self.srla
                return self.find_trivial_epsilon(epsilon_middle, epsilon2)


    def find_edge_epsilon(self, epsilon1, epsilon2):
        """ Finds the first epsilon for which we have a regular partition.
        :returns: float epsilon
        """
        step = (epsilon2 - epsilon1)/2.0
        if step < self.min_step:
            return epsilon2
        else:
            epsilon_middle = epsilon1 + step
            regular, k, classes, sze_idx, reg_list, nirr= self.run_alg(epsilon_middle)
            if self.verbose:
                print(f"    |{epsilon1:.6f}-----{epsilon_middle:.6f}------{epsilon2:.6f}| {k} {regular}")
            if regular:
                #if k>self.min_k :
                    #self.epsilons.append(epsilon_middle)
                del self.srla
                return self.find_edge_epsilon(epsilon1, epsilon_middle)
            else:
                del self.srla
                return self.find_edge_epsilon(epsilon_middle, epsilon2)


    def find_bounds(self):
        """ Finds the bounding epsilons and set up the range where to search
        :returns: the two bounds found
        """
        if self.bounds:
            epsilon1 = self.bounds[0]
            epsilon2 = self.bounds[1]
        else:
            if self.verbose:
                print("     Finding trivial epsilon...")
            epsilon2 = self.find_trivial_epsilon(0, 1)
            if self.verbose:
                print(f"    Trivial epsilon candidate: {epsilon2:.6f}")
                print("    Finding edge epsilon...")
            epsilon1 = self.find_edge_epsilon(0, epsilon2)
            if self.verbose:
                print(f"    Edge epsilon candidate: {epsilon1:.6f}")
        self.bounds = [epsilon1, epsilon2]
        self.epsilons.append(epsilon1)
        # Try self.tries different epsilons inside the bounds
        offs = (epsilon2 - epsilon1) / self.tries
        for i in range(1, self.tries+1):
            self.epsilons.append(epsilon1 + (i*offs))

        return self.bounds


    def find_partitions(self):
        """ Find partitions of the graph
        :returns: a dictionary
        """
        self.k_e_c_i= {}
        for epsilon in self.epsilons:
            regular, k, classes, sze_idx, reg_list , nirr= self.run_alg(epsilon)

            if self.verbose:
                print(f"    {epsilon:.6f} {k} {regular} {sze_idx:.4f}")

            if regular:
                if k in self.k_e_c_i:
                    if epsilon < self.k_e_c_i[k][0]: #sze_idx or epsilon?
                        self.k_e_c_i[k] = (epsilon, classes, sze_idx, reg_list, nirr)
                else:
                    self.k_e_c_i[k] = (epsilon, classes, sze_idx, reg_list, nirr)

            elif self.fast_search and not regular and k>self.min_k:
                return self.k_e_c_i

        return self.k_e_c_i

    def find_bestL2_partitions(self):
        min_l2 = 2
        min_partition = None
        #self.epsilons.append(0.218750)
        for epsilon in self.epsilons:

            regular, k, classes, sze_idx, reg_list , nirr= self.run_alg(epsilon)

            if regular:

                sze_rec = self.reconstruct_mat(0, classes, k, reg_list)
                dist2 = self.L2_metric_GT(sze_rec)
                dist1 = self.L1_metric_GT(sze_rec)

                if dist2 < min_l2:
                    min_l2 = dist2
                    min_partition = (epsilon, classes, sze_idx, reg_list, sze_rec, k, dist2, dist1)
                    print(f"    Distance: L1wrtGT :{dist1:.4f}  L2wrtGT :{dist2:.4f}")
                    print("     New BEST!")

        return min_partition


    def find_bestL2_partitions_threshold(self, density):
        min_l2 = 2
        min_partition = None
        #self.epsilons.append(0.218750)
        for epsilon in self.epsilons:
            print(f"eps: {epsilon:.4f}")

            regular, k, classes, sze_idx, reg_list , nirr= self.run_alg(epsilon)

            if regular and k>30:
                #print(f"    {epsilon:.6f} {k} {regular} {sze_idx:.4f}")

                t_max = -1
                t_sze_rec = None
                t_dist2_min = 10
                print(f"density:{density}")
                for t in np.arange(0, density, 0.0005):
                    sze_rec = self.reconstruct_mat(t, classes, k, reg_list)
                    dist2 = self.L2_metric_GT(sze_rec)
                    print(f"t:{t:.2f} - L2:{dist2:.4f}")

                    if dist2 < t_dist2_min:
                        t_max = t
                        t_sze_rec = sze_rec

                dist2 = self.L2_metric_GT(t_sze_rec)
                dist1 = self.L1_metric_GT(t_sze_rec)

                if dist2 < min_l2:
                    min_l2 = dist2
                    min_partition = (epsilon, classes, sze_idx, reg_list, t_sze_rec, k, dist2, dist1)
                    print(f"    Distance: L1wrtGT :{dist1:.4f}  L2wrtGT :{dist2:.4f}")
                    print("     New BEST!")

        return min_partition


    def thresholds_analysis(self, classes, k, reg_list, thresholds, measure):
        """ Performs threshold analysis with a given measure
        :param classes: the reduced array
        :param k: the cardinality of the patition
        :param measure: the measure to use
        :returns: the measures calculated with sze_rec
        """
        self.measures = []
        for thresh in thresholds:
            sze_rec = self.reconstruct_mat(thresh, classes, k, reg_list)
            res = measure(sze_rec)

            #self.plot_graphs(self.G, sze_rec, thresh)

            if self.verbose:
                print(f"    {res:.5f}")
            self.measures.append(res)

        return self.measures



    def reconstruct_mat_trick(self, thresh, classes, k, regularity_list):
        reconstructed_mat = np.zeros((self.G.shape[0], self.G.shape[0]), dtype='float32')

        # Implements indensity information preservation

        for i in range(1, k+1):
            print(i)
            reconstructed_mat = np.zeros((self.G.shape[0], self.G.shape[0]), dtype='float32')
            for c in range(i, i+1):
                indices_c = np.where(classes == c)[0]
                n = len(indices_c)
                max_edges = (n*(n-1))/2
                n_edges = np.tril(self.G[np.ix_(indices_c, indices_c)], -1).sum()
                indensity = n_edges / max_edges
                reconstructed_mat[np.ix_(indices_c, indices_c)] = 1#indensity

            plt.imshow(reconstructed_mat)
            plt.savefig(f"/tmp/classe_{i}.png")

        np.fill_diagonal(reconstructed_mat, 0.0)
        return reconstructed_mat

    def reconstruct_mat(self, thresh, classes, k, regularity_list):
        """ Reconstruct the original matrix from a reduced one.
        :param thres: the edge threshold if the density between two pairs is over it we put an edge
        :param classes: the reduced graph expressed as an array
        :return: a numpy matrix of the size of GT

        -----
        [TODO BUG] wrong implementation on weighted case indensity preservation

        """
        reconstructed_mat = np.zeros((self.G.shape[0], self.G.shape[0]), dtype='float32')
        for r in range(2, k + 1):
            r_nodes = np.where(classes == r)[0]

            for s in range(1, r) if not self.drop_edges_between_irregular_pairs else regularity_list[r - 2]:
                s += 1
                s_nodes = np.where(classes == s)[0]
                bip_sim_mat = self.G[np.ix_(r_nodes, s_nodes)]
                n = bip_sim_mat.shape[0]
                bip_density = bip_sim_mat.sum() / (n ** 2.0)

                # Put edges if above threshold
                if bip_density > thresh:
                    if self.is_weighted:
                        reconstructed_mat[np.ix_(r_nodes, s_nodes)] = reconstructed_mat[np.ix_(s_nodes, r_nodes)] = bip_density
                    else:
                        reconstructed_mat[np.ix_(r_nodes, s_nodes)] = reconstructed_mat[np.ix_(s_nodes, r_nodes)] = 1

        # Implements indensity information preservation
        if self.indensity_preservation:
            for c in range(1, k+1):
                indices_c = np.where(classes == c)[0]
                n = len(indices_c)
                max_edges = (n*(n-1))/2
                n_edges = np.tril(self.G[np.ix_(indices_c, indices_c)], -1).sum()
                indensity = n_edges / max_edges
                if np.random.uniform(0,1,1) <= indensity:
                    if self.is_weighted:
                        # [TODO BUG] wrong implementation
                        reconstructed_mat[np.ix_(indices_c, indices_c)] = indensity
                    else:
                        erg = np.tril(np.random.random((n, n)) <= indensity, -1).astype('float32')
                        erg += erg.T
                        reconstructed_mat[np.ix_(indices_c, indices_c)] = erg

        np.fill_diagonal(reconstructed_mat, 0.0)
        return reconstructed_mat

    def generate_reduced_sim_mat(self, k, epsilon, classes, regularity_list):
        """
        generate the similarity matrix of the current classes
        :return sim_mat: the reduced similarity matrix
        """
        reduced_sim_mat = np.zeros((k, k), dtype='float32')

        for r in range(2, k + 1):

            for s in (range(1, r) if not self.drop_edges_between_irregular_pairs else regularity_list[r - 2]):

                # Classes indices w.r.t. original graph uint16 from 0 to 65535
                s_indices = np.where(classes == s)[0].astype('uint16')
                r_indices = np.where(classes == r)[0].astype('uint16')

                if self.is_weighted:

                    adj_mat = (self.G > 0).astype("int8")
                    bip_adj_mat = adj_mat[np.ix_(s_indices, r_indices)]

                else:
                    bip_adj_mat = self.G[np.ix_(s_indices, r_indices)]

                classes_n = bip_adj_mat.shape[0]
                bip_density = bip_adj_mat.sum() / (classes_n ** 2.0)
                reduced_sim_mat[r - 1, s - 1] = reduced_sim_mat[s - 1, r - 1] = bip_density

        return reduced_sim_mat

    #################
    #### Metrics ####
    #################


    def KLdivergence_metric_GT(self, graph):
        """ Computes thhe kulback liebeler divergence
        :param graph: np.array() reconstructed graph
        :returns: np.array(float64) feature vector of measures
        """
        p1 = self.GT.sum(0)
        p2 = graph.sum(0)

        e1 = spst.entropy(p1, p2)
        e2 = spst.entropy(p2, p1)

        return e1, e2

    def KLdivergence_metric(self, graph):
        """ Computes thhe kulback liebeler divergence
        :param graph: np.array() reconstructed graph
        :returns: np.array(float64) feature vector of measures
        """
        p1 = self.G.sum(0)
        p2 = graph.sum(0)

        e1 = spst.entropy(p1, p2)
        e2 = spst.entropy(p2, p1)

        return e1, e2


    def KVS_metric(self, graph):
        """ Implements Knn Voting System to calculate if the labeling is correct.
        :param graph: reconstructed graph
        :returns: adjusted random score
        """
        n = len(self.labels)

        max_ars = -10

        for k in [5, 7, 10]:
            candidates = np.zeros(n, dtype='uint32')
            i = 0
            for row in graph:
                max_k_idxs = row.argsort()[-k:]
                aux = row[max_k_idxs] > 0
                k_indices = max_k_idxs[aux]

                if len(k_indices) == 0:
                    k_indices = row.argsort()[-1:]

                #candidate_lbl = np.bincount(self.labels[k_indices].astype(int)).argmax()
                candidate_lbl = np.bincount(self.labels[k_indices]).argmax()
                candidates[i] = candidate_lbl
                i += 1

            ars = metrics.adjusted_rand_score(self.labels, candidates)
            if ars > max_ars:
                max_k = k
                max_ars = ars


        return max_ars

    def RED_KVS_metric(self, graph, labels):
        """ Implements Knn Voting System to calculate if the labeling is correct.
        :param graph: reconstructed graph
        :returns: adjusted random score
        """
        n = graph.shape[0]

        max_ars = -10

        for k in [3, 5, 7, 9, 10, 15]:
            candidates = np.zeros(n, dtype='int16')
            i = 0
            for row in graph:
                max_k_idxs = row.argsort()[-k:]
                aux = row[max_k_idxs] > 0
                k_classes = max_k_idxs[aux]

                if len(k_classes) == 0:
                    k_classes = row.argsort()[-1:]

                candidate_lbl = np.bincount(labels[k_classes]).argmax()
                candidates[i] = candidate_lbl
                i += 1

            ars = metrics.adjusted_rand_score(self.labels, candidates)
            if ars > max_ars:
                max_k = k
                max_ars = ars

        return max_ars

    def L2_metric(self, graph):
        """ Compute the normalized L2 distance between two matrices
        :param graph: np.array, reconstructed graph
        :returns: float, L2 norm
        """
        return np.linalg.norm(self.G-graph)/self.G.shape[0]


    def L1_metric(self, graph):
        """ Compute the normalized L1 distance between two matrices
        :param graph: np.array, reconstructed graph
        :returns: float, L2 norm
        """
        return (np.abs(graph - self.G)/self.G.shape[0]**2).sum()

    def L2_metric_GT(self, graph):
        return np.linalg.norm(self.GT-graph)/self.GT.shape[0]


    def L1_metric_GT(self, graph):
        return (np.abs(graph - self.GT)/self.GT.shape[0]**2).sum()


    def ACT_metric(self, graph):
        """ Compute Amplitude Commute Time with a Matlab engine request, it does not
        apply the Gaussian Kernel
        :param graph: reconstructed graph
        :returns: L2 distance between GT_t and abs(NG_t-1)
        """
        if not self.eng:
            print("[+] Starting Matlab engine ...")
            self.eng = matlab.engine.start_matlab()
            self.eng.addpath(r'./matlab_code',nargout=0)

        mat_GT = matlab.double(self.GT.tolist())
        mat_NG = matlab.double(graph.tolist())
        res = self.eng.get_ACT(mat_NG, mat_GT)

        return res


    def NGU_metric(self, graph):
        """ Compute Nguyen metric
        :param graph: reconstructed graph
        :returns: accuracy of Nguyen paper
        """
        if not self.eng:
            print("[+] Starting Matlab engine ...")
            self.eng = matlab.engine.start_matlab()
            self.eng.addpath(r'./matlab_code',nargout=0)

        mat_NG = matlab.double(self.G.tolist())
        mat_sze_rec = matlab.double(graph.tolist())

        self.eng.myconnect(mat_sze_rec)
        res = self.eng.kmedoidsAccdat(mat_sze_rec, matlab.int32(np.vstack(self.labels).tolist()), np.unique(self.labels).size)

        return res
 
    def DS_metric(self, graph):
        """ Implements Dominant Set Voting System to calculate if the labeling is correct.
        :param graph: reconstructed graph
        :returns: adjusted random score
        """

        candidates = self.dominant_sets(graph)

        ars = metrics.adjusted_rand_score(self.labels, candidates)

        return ars

    def replicator(self, A, x, inds, tol, max_iter):
        error = tol + 1.0
        count = 0
        while error > tol and count < max_iter:
            x_old = np.copy(x)
            for i in inds:
                x[i] = x_old[i] * (A[i] @ x_old)
            x /= np.sum(x)
            error = np.linalg.norm(x - x_old)
            count += 1
        return x


    def dominant_sets(self, graph_mat, max_k=4, tol=1e-5, max_iter=100):
        graph_cardinality = graph_mat.shape[0]
        if max_k == 0:
            max_k = graph_cardinality
        clusters = np.zeros(graph_cardinality)
        already_clustered = np.full(graph_cardinality, False, dtype=np.bool)

        for k in range(max_k):
            if graph_cardinality - already_clustered.sum() <= ceil(0.05 * graph_cardinality):
                break
            # 1000 is added to obtain more similar values when x is normalized
            # x = np.random.random_sample(graph_cardinality) + 1000.0
            x = np.full(graph_cardinality, 1.0)
            x[already_clustered] = 0.0
            x /= x.sum()

            y = self.replicator(graph_mat, x, np.where(~already_clustered)[0], tol, max_iter)
            cluster = np.where(y >= 1.0 / (graph_cardinality * 1.5))[0]
            already_clustered[cluster] = True
            clusters[cluster] = k
        clusters[~already_clustered] = k
        return clusters
