import numpy as np
import scipy.sparse.linalg
import math
import sys
import ipdb


def alon1(self, cl_pair):
    """
    verify the first condition of Alon algorithm (regularity of pair)
    :param cl_pair: the bipartite graph to be checked
    :return: True if the condition is verified, False otherwise
    :return: A list of two empty lists representing the empty certificates
    :return: A list of two empty lists representing the empty complements
    """
    return cl_pair.bip_avg_deg < (self.epsilon ** 3.0) * cl_pair.n, [[], []], [[], []]



def get_s_r_degrees(self, s, r):
    """ Given two classes it returns a degree vector (indicator vector) where the degrees
    have been calculated with respecto to each other set.
    :param s: int, class s
    :param r: int, class r
    :returns: np.array, degree vector
    """

    s_r_degs = np.zeros(len(self.degrees), dtype='int16')

    # Gets the indices of elements which are part of class s, then r
    s_indices = np.where(self.classes == s)[0]
    r_indices = np.where(self.classes == r)[0]

    # Calculates the degree and assigns it
    s_r_degs[s_indices] = self.adj_mat[np.ix_(s_indices, r_indices)].sum(1)
    s_r_degs[r_indices] = self.adj_mat[np.ix_(r_indices, s_indices)].sum(1)

    return s_r_degs


def alon2(self, cl_pair):
    """ Verifies the third condition of Alon algorithm (irregularity of pair) and return the pair's certificate and
    complement in case of irregularity
    :param cl_pair: the bipartite graph to be checked
    :return: True if the condition is verified, False otherwise
    """

    #ipdb.set_trace()

    # Gets the vector of degrees of nodes of class s wrt class r
    s_degrees = get_s_r_degrees(self, cl_pair.s, cl_pair.r)
    s_indices = np.where(self.classes == cl_pair.s)[0]
    s_degrees = s_degrees[s_indices]

    deviated_nodes_mask = np.abs(s_degrees - cl_pair.bip_avg_deg) >= (self.epsilon ** 4.0) * cl_pair.n

    if deviated_nodes_mask.sum() > (1/8 * self.epsilon**4 * cl_pair.n):
        # [TODO] Heuristic? Zip?
        s_certs = s_indices[deviated_nodes_mask]
        s_compls = np.setdiff1d(s_indices, s_certs)

        # Takes all the indices of class r which are connected to s_certs
        r_indices = np.where(self.classes == cl_pair.r)[0] #[TODO] optimiz, reuse structure
        b_mask = self.adj_mat[np.ix_(s_certs, r_indices)] > 0
        b_mask = b_mask.any(0)

        r_certs = r_indices[b_mask]
        r_compls = np.setdiff1d(r_indices, r_certs)

        is_irregular = True
        return is_irregular, [r_certs.tolist(), s_certs.tolist()], [r_compls.tolist(), s_compls.tolist()]
    else:
        is_irregular = False
        return is_irregular, [[], []], [[], []]


def alon3(self, cl_pair):
    """ Verifies the third condition of Alon algorithm (irregularity of pair) and return the pair's certificate and
    complement in case of irregularity
    :param cl_pair: the bipartite graph to be checked
    :return: True if the condition is verified, False otherwise
    """
    is_irregular = False

    nh_dev_mat, aux = cl_pair.neighbourhood_deviation_matrix()

    # Gets the vector of degrees of nodes of class s wrt class r
    s_degrees = get_s_r_degrees(self, cl_pair.s, cl_pair.r)
    s_indices = np.where(self.classes == cl_pair.s)[0]
    s_degrees = s_degrees[s_indices]

    yp_filter = cl_pair.find_Yp(s_degrees, s_indices)

    if yp_filter.size == 0:
        is_irregular = True
        return is_irregular, [[], []], [[], []]

    s_certs, y0 = cl_pair.compute_y0(nh_dev_mat, s_indices, yp_filter)

    if s_certs is None:
        is_irregular = False
        return is_irregular, [[], []], [[], []]
    else:
        assert np.array_equal(np.intersect1d(s_certs, s_indices), s_certs) == True, "cert_is not subset of s_indices"
        assert (y0 in s_indices) == True, "y0 not in s_indices"

        is_irregular = True
        r_indices = np.where(self.classes == cl_pair.r)[0] #[TODO] optimiz
        b_mask = self.adj_mat[np.ix_(np.array([y0]), r_indices)] > 0
        r_certs = r_indices[b_mask[0]]
        assert np.array_equal(np.intersect1d(r_certs, r_indices), r_certs) == True, "cert_is not subset of s_indices"

        # [BUG] cannot do set(s_indices) - set(s_certs)
        s_compls = np.setdiff1d(s_indices, s_certs)
        r_compls = np.setdiff1d(r_indices, r_certs)
        assert s_compls.size + s_certs.size == self.classes_cardinality, "Wrong cardinality"
        assert r_compls.size + r_certs.size == self.classes_cardinality, "Wrong cardinality"

        return is_irregular, [r_certs.tolist(), s_certs.tolist()], [r_compls.tolist(), s_compls.tolist()]

