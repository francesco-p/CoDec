import numpy as np
import ipdb


class ClassesPair:

    def __init__(self, adj_mat, classes, r, s, epsilon):

        # Classes id
        self.r = r
        self.s = s

        # Classes indices w.r.t. original graph uint16 from 0 to 65535
        self.s_indices = np.where(classes == self.s)[0].astype('uint16')
        self.r_indices = np.where(classes == self.r)[0].astype('uint16')

        # Bipartite adjacency matrix, we've lost the indices w.r.t. adj_mat, inherits dtype of adj_mat
        self.bip_adj_mat = adj_mat[np.ix_(self.s_indices, self.r_indices)]

        # Cardinality of the classes
        self.classes_n = self.bip_adj_mat.shape[0]

        # Bipartite average degree
        self.bip_avg_deg = (self.bip_adj_mat.sum(0) + self.bip_adj_mat.sum(1)).sum() / (2.0 * self.classes_n)

        # Compute the density of a bipartite graph as the sum of the edges over the number of all possible edges in the bipartite graph
        self.bip_density = self.bip_adj_mat.sum() / (self.classes_n ** 2.0)

        # Current epsilon used
        self.epsilon = epsilon

        # Degree vector (indicator vector) where the degrees have been calculated with respect to each other set.
        self.s_r_degrees = np.zeros(len(classes), dtype='uint16')

        # Calculates the degree and assigns it
        self.s_r_degrees[self.s_indices] = adj_mat[np.ix_(self.s_indices, self.r_indices)].sum(1)
        self.s_r_degrees[self.r_indices] = adj_mat[np.ix_(self.r_indices, self.s_indices)].sum(1)


    def neighbourhood_deviation_matrix(self):

        mat = self.bip_adj_mat.T @ self.bip_adj_mat
        mat = mat.astype('float32')
        mat -= (self.bip_avg_deg ** 2.0) / self.classes_n
        return mat


    def find_Yp(self, bip_degrees, s_indices):
        """ Find a subset of s_indices which will create the Y', it could return an empty array
        :param bip_degrees: np.array(int32) array of the degrees of s nodes w.r.t. class r
        :param s_indices: np.array(int32) array of the indices of class s
        :return: np.array(int32) subset of indices of class s 
        """
        mask = np.abs(bip_degrees - self.bip_avg_deg) < ((self.epsilon ** 4.0) * self.classes_n)
        yp_i = np.where(mask == True)[0]
        return yp_i


    def compute_y0(self, nh_dev_mat, s_indices, yp_i):
        """ Finds y0 index node and certificates indices 
        :param nh_dev_mat: np.array((s.size, s.size), dtype='float32') neighbourhood deviation matrix of class s
        :param s_indices: np.array(self.classes_cardinality, dtype='float32') indices of the nodes of class s
        :param yp_i: np.array(dtype='float32') these are indices of the rows to be filtered in the nh_dev_mat
        :return: tuple cert_s which is a np.array(float32) subset of s_indices if it possible, None otherwise

        [TODO] : 
            - why yp_i float32?
            - type of cert_s is float32?
        """

        # Create rectancular matrix to create |y'| sets
        rect_mat = nh_dev_mat[yp_i]

        # Check which set have the best neighbour deviation
        boolean_matrix = rect_mat > (2 * self.epsilon**4 * self.classes_n)
        cardinality_by0s = boolean_matrix.sum(1)

        # Select the best set
        y0_idx = np.argmax(cardinality_by0s)
        aux = yp_i[y0_idx]

        # Gets the y0 index
        y0 = s_indices[aux]

        if cardinality_by0s[y0_idx] > (self.epsilon**4 * self.classes_n / 4.0):
            cert_s = s_indices[boolean_matrix[y0_idx]]
            return cert_s, y0
        else:
            return None, y0


class WeightedClassesPair:
    def __init__(self, sim_mat, adj_mat, classes, r, s, epsilon):
        pass

"""
class WeightedClassesPair:
    bip_sim_mat = np.empty((0, 0), dtype='float32')
    bip_adj_mat = np.empty((0, 0), dtype='int8')
    r = s = -1
    n = 0
    index_map = np.empty((0, 0))
    bip_avg_deg = 0.0
    bip_density = 0.0
    epsilon = 0.0

    def __init__(self, sim_mat, adj_mat, classes, r, s, epsilon):
        self.r = r
        self.s = s
        self.index_map = np.where(classes == r)[0]
        self.index_map = np.vstack((self.index_map, np.where(classes == s)[0]))
        self.bip_sim_mat = sim_mat[np.ix_(self.index_map[0], self.index_map[1])]
        self.bip_adj_mat = adj_mat[np.ix_(self.index_map[0], self.index_map[1])]
        self.n = self.bip_sim_mat.shape[0]
        self.bip_avg_deg = self.bip_avg_degree()
        self.bip_density = self.compute_bip_density()
        self.epsilon = epsilon

    def bip_avg_degree(self):
        return (self.bip_sim_mat.sum(0) + self.bip_sim_mat.sum(1)).sum() / (2.0 * self.n)

    def compute_bip_density(self):
        return self.bip_sim_mat.sum() / (self.n ** 2.0)

    def classes_vertices_degrees(self):
        c_v_degs = np.sum(self.bip_adj_mat, 0)
        c_v_degs = np.vstack((c_v_degs, np.sum(self.bip_adj_mat, 1)))
        return c_v_degs

    # def neighbourhood_matrix(self, transpose_first=True):
    #     if transpose_first:
    #         return self.bip_adj_mat.T @ self.bip_adj_mat
    #     else:
    #         return self.bip_adj_mat @ self.bip_adj_mat.T
    #
    # def neighbourhood_deviation_matrix(self, nh_mat):
    #     return nh_mat - ((self.bip_avg_deg ** 2.0) / self.n)

    def neighbourhood_deviation_matrix(self, transpose_first=True):
        if transpose_first:
            mat = self.bip_adj_mat.T @ self.bip_adj_mat
        else:
            mat = self.bip_adj_mat @ self.bip_adj_mat.T
        rs_degrees = np.diag(mat)
        mat -= (self.bip_avg_deg ** 2.0) / self.n
        return mat, rs_degrees

    def find_Y(self, nh_dev_mat):
        inner_sums = nh_dev_mat.sum(1) - np.diag(nh_dev_mat)
        inner_sums_indices = np.argsort(inner_sums)[::-1]
        y_card_thresh = int((self.epsilon * self.n) + 1)
        outer_sum = inner_sums[inner_sums_indices[0:(y_card_thresh - 1)]].sum()

        for i in range(y_card_thresh, self.n):
            outer_sum += inner_sums[inner_sums_indices[i]]
            sigma_y = outer_sum / (i ** 2.0)
            if sigma_y >= ((self.epsilon ** 3.0) / 2.0) * self.n:
                return inner_sums_indices[0:i]
        return np.array([])

    def find_Yp(self, degrees, Y_indices):

        return Y_indices[np.abs(degrees - self.bip_avg_deg) < ((self.epsilon ** 4.0) * self.n)]

    def get_s_r_degrees(self):

        s_r_degs = np.zeros(len(self.degrees), dtype='int16')

        # Gets the indices of elements which are part of class s, then r
        s_indices = np.where(self.classes == self.s)[0]
        r_indices = np.where(self.classes == self.r)[0]

        # Calculates the degree and assigns it
        s_r_degs[s_indices] = self.adj_mat[np.ix_(s_indices, r_indices)].sum(1)
        s_r_degs[r_indices] = self.adj_mat[np.ix_(r_indices, s_indices)].sum(1)

        return s_r_degs

    def compute_y0(self, nh_dev_mat, Y_indices, Yp_indices):
        sums = np.full((self.n,), -np.inf)
        for i in Yp_indices:
            sums[i] = 0
            for j in list(set(Y_indices) - set(Yp_indices)):
                sums[i] += nh_dev_mat[i, j]
        return np.argmax(sums)

    def find_s_cert_and_compl(self, nh_dev_mat, y0, Yp_indices):
        outliers_in_s = set(np.where(nh_dev_mat[y0, :] > 2.0 * (self.epsilon ** 4.0) * self.n)[0])
        outliers_in_Yp = list(set(Yp_indices) & outliers_in_s)
        cert = list(self.index_map[1][outliers_in_Yp])
        compl = [self.index_map[1][i] for i in range(self.n) if i not in outliers_in_Yp]
        return cert, compl

    def find_r_cert_and_compl(self, y0):
        indices = np.where(self.bip_adj_mat[:, y0] > 0.0)[0]
        cert = list(self.index_map[0][indices])
        compl = [self.index_map[0][i] for i in range(self.n) if i not in indices]
        return cert, compl
"""
