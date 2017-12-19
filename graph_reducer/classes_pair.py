import numpy as np
import ipdb


class ClassesPair:
    bip_adj_mat = np.empty((0, 0), dtype='int8')
    """The bipartite adjacency matrix. Given a bipartite graph with classes r and s, the rows of this matrix represent
       the nodes in r, while the columns the nodes in s"""
    r = s = -1
    """The classes composing the bipartite graph"""
    n = 0
    """the cardinality of a class"""
    index_map = np.empty((0, 0))
    """A mapping from the bipartite adjacency matrix nodes to the adjacency matrix ones"""
    bip_avg_deg = 0
    """the average degree of the graph"""
    bip_density = 0
    """the average density of the graph"""
    epsilon = 0.0
    """the epsilon parameter"""

    def __init__(self, adj_mat, classes, r, s, epsilon):
        self.r = r
        self.s = s
        # [TODO] optimization: why using index_map? and vstack? to check the cardinality?
        self.index_map = np.where(classes == r)[0]
        self.index_map = np.vstack((self.index_map, np.where(classes == s)[0]))
        self.bip_adj_mat = adj_mat[np.ix_(self.index_map[0], self.index_map[1])]
        # [TODO] question: wouldn't it be = to self.classes_cardinality?
        self.n = self.bip_adj_mat.shape[0]
        self.bip_avg_deg = self.bip_avg_degree()
        self.bip_density = self.compute_bip_density()
        self.epsilon = epsilon

    def bip_avg_degree(self):
        """
        compute the average degree of the bipartite graph
        :return the average degree
        """
        return (self.bip_adj_mat.sum(0) + self.bip_adj_mat.sum(1)).sum() / (2.0 * self.n)

    def compute_bip_density(self):
        """
        compute the density of a bipartite graph as the sum of the edges over the number of all possible edges in the
        bipartite graph
        :return the density
        """
        # [TODO] optimization: does sum() already return a float?
        #ipdb.set_trace()
        return float(self.bip_adj_mat.sum()) / (self.n ** 2.0)

    def classes_vertices_degrees(self):
        """
        compute the degree of all vertices in the bipartite graph
        :return a (n,) numpy array containing the degree of each vertex
        """
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
        # [TODO] can we do it in gpu with cuBLAS? :)
        if transpose_first:
            mat = self.bip_adj_mat.T @ self.bip_adj_mat
        else:
            mat = self.bip_adj_mat @ self.bip_adj_mat.T
        rs_degrees = np.diag(mat)
        mat -= (self.bip_avg_deg ** 2.0) / self.n
        return mat, rs_degrees

    def get_s_r_degrees(self):
        """ Given two classes it returns a degree vector (indicator vector) where the degrees
        have been calculated with respecto to each other set.
        :param s: int, class s
        :param r: int, class r
        :returns: np.array, degree vector
        """

        s_r_degs = np.zeros(len(self.degrees), dtype='int16')

        # Gets the indices of elements which are part of class s, then r
        s_indices = np.where(self.classes == self.s)[0]
        r_indices = np.where(self.classes == self.r)[0]

        # Calculates the degree and assigns it
        s_r_degs[s_indices] = self.adj_mat[np.ix_(s_indices, r_indices)].sum(1)
        s_r_degs[r_indices] = self.adj_mat[np.ix_(r_indices, s_indices)].sum(1)

        return s_r_degs

    def find_Y(self, nh_dev_mat):
        inner_sums = nh_dev_mat.sum(1) - np.diag(nh_dev_mat)
        inner_sums_indices = np.argsort(inner_sums)[::-1]
        y_card_thresh = int((self.epsilon * self.n) + 1)
        outer_sum = inner_sums[inner_sums_indices[0:(y_card_thresh - 1)]].sum()

        for i in range(y_card_thresh, self.n):
            outer_sum += inner_sums[inner_sums_indices[i]]
            sigma_y = outer_sum / (i ** 2.0)
            # print "sigma_y = " + str(sigma_y)
            if sigma_y >= ((self.epsilon ** 3.0) / 2.0) * self.n:
                return inner_sums_indices[0:i]
        return np.array([])


    def find_Yp(self, bip_degrees, s_indices):
        # [TODO ASAP BUG] indices of s_indices
        #return s_indices[np.abs(bip_degrees - self.bip_avg_deg) < ((self.epsilon ** 4.0) * self.n)]
        mask = np.abs(bip_degrees - self.bip_avg_deg) < ((self.epsilon ** 4.0) * self.n)
        return np.where(mask == True)


    def compute_y0(self, nh_dev_mat, s_indices, yp_indices):
        """ Find y0 and certificates if it is the case """

        rect_mat = nh_dev_mat[yp_indices]
        boolean_matrix = rect_mat > (2 * self.epsilon**4 * self.n)
        cardinality_by0s = boolean_matrix.sum(1)

        y0 = np.argmax(cardinality_by0s)

        if cardinality_by0s[y0] > (self.epsilon**4 * self.n / 4.0):
            cert_s = s_indices[boolean_matrix[y0]]
            return cert_s, y0
        else:
            return None, y0


    def old_compute_y0(self, nh_dev_mat, Y_indices, Yp_indices):
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
        indices = np.where(self.bip_adj_mat[:, y0] > 0)[0]
        cert = list(self.index_map[0][indices])
        compl = [self.index_map[0][i] for i in range(self.n) if i not in indices]
        return cert, compl


class WeightedClassesPair:
    bip_sim_mat = np.empty((0, 0), dtype='float32')
    """The bipartite similarity matrix. Given a bipartite graph with classes r and s, the rows of this matrix represent
       the nodes in r, while the columns the nodes in s."""
    bip_adj_mat = np.empty((0, 0), dtype='int8')
    """The bipartite adjacency matrix. Given a bipartite graph with classes r and s, the rows of this matrix represent
       the nodes in r, while the columns the nodes in s"""
    r = s = -1
    """The classes composing the bipartite graph"""
    n = 0
    """the cardinality of a class"""
    index_map = np.empty((0, 0))
    """A mapping from the bipartite adjacency matrix nodes to the adjacency matrix ones"""
    bip_avg_deg = 0.0
    """the average degree of the graph"""
    bip_density = 0.0
    """the average density of the graph"""
    epsilon = 0.0
    """the epsilon parameter"""

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
        """
        compute the average degree of the bipartite graph
        :return the average degree
        """
        return (self.bip_sim_mat.sum(0) + self.bip_sim_mat.sum(1)).sum() / (2.0 * self.n)

    def compute_bip_density(self):
        """
        compute the density of a bipartite graph as the sum of the edges over the number of all possible edges in the
        bipartite graph
        :return the density
        """
        return self.bip_sim_mat.sum() / (self.n ** 2.0)

    def classes_vertices_degrees(self):
        """
        compute the degree of all vertices in the bipartite graph
        :return a (n,) numpy array containing the degree of each vertex
        """
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
        """ Given two classes it returns a degree vector (indicator vector) where the degrees
        have been calculated with respecto to each other set.
        :param s: int, class s
        :param r: int, class r
        :returns: np.array, degree vector
        """

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
