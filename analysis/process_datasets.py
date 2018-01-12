import scipy.io as sp
import numpy as np
#import ipdb
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys


def check_validity(G, n, desired_d):
    """ Checks dimension and data type of the graph to reduce space consumption
    Checks that the density is in desired_density +/- tolerance
    """
    assert G.shape[0] == n, "Dimension is not correct"
    assert G.dtype == 'int8', "Data type of the graph is not correct"
    tolerance = 0.015
    desired_d = round(desired_d, 2)
    graph_d = round(density(G), 2)
    assert desired_d-tolerance < graph_d and graph_d < desired_d+tolerance, "Density is not correct"


def synthetic_graph(n, d):
    """ Generate a n,n graph with a given density d
    :param d: float density of the graph
    :return: np.array((n, n), dtype='float32') graph with density d
    """
    G = np.tril(np.random.random((n, n)) <= d, -1).astype('int8')
    return G + G.T

def synthethic_grid_graph(n):
    """ Generate a n,n mesh graph
    :param n: int number of nodes of the graph
    :return: np.array((n, n), dtype='float32') graph with density d
    """

    l = [x for x in range(0,n+1)]

    groups = zip(l[::5], l[1::5], l[2::5], l[3::5], l[4::5])
    groups = list(groups)

    a = np.array(groups[::2]).flatten()
    b = np.array(groups[1::2]).flatten()

    mat = np.ones((n,n))

    mat[np.ix_(a, b)] = 0
    mat += mat.T
    mat[np.where(mat == 1)] = 0
    mat /= 2

    np.fill_diagonal(mat, 0)

    return mat

def density(G, weighted=False):
    """ Calculates the density of a synthetic graph
    :param G: np.array((n, n)) graph
    :param weighted: bool flag to discriminate weighted case
    :return: float density of the graph
    """
    n = G.shape[0]

    if weighted:
        return G.sum() / (n ** 2)

    e = np.where(G == 1)[0].size / 2
    return e / ((n*(n-1))/2)

def search_dset(filename, synth=False):
    """ Search for a .npz file into data/npz/ folder then if exists it returns the dictionary with NG, GT, bounds
    :param dataset: of the dataset
    :param sigma: sigma of the gaussian kernel
    :returns: None if no file or the dictionary
    """
    if synth:
        loaded = np.load(filename)
        #print(f"#### [+] {n}_{d:.2f}_{dset_id}.npy correctly loaded ####")
        data = {}
        data['G'] = loaded['G']
        data['GT'] = []
        data['bounds'] = loaded['bounds']
        data['labels'] = []
        return data
    else:
        path = "data/npz/"
        for f in os.listdir(path):
            if f == filename+".npz":
                return np.load(path+f)
        raise FileNotFoundError(f"{path}{filename}.npz")


def synthetic_graph(n, d):
    """ Generate a n,n graph with a given density d
    :param d: float density of the graph
    :return: np.array((n, n), dtype='float32') graph with density d
    """
    G = np.tril(np.random.random((n, n)) < d, -1).astype('int8')
    return G + G.T


def density(G):
    """ Check density of a synthetic graph
    :param G: np.array((n, n), dtype='int8') graph
    :return: float density of the graph
    """
    n = G.shape[0]
    e = np.where(G == 1)[0].size / 2
    return e / ((n*(n-1))/2)


def graph_from_points(x, sigma, to_remove=0):
    """ Generates a graph (weighted graph) from a set of points x (nxd) and a sigma decay
    :param x: a numpy matrix of n times d dimension
    :param sigma: a sigma for the gaussian kernel
    :param to_remove: imbalances the last cluster
    :return: a weighted symmetric graph
    """

    n = x.shape[0]
    n -= to_remove
    w_graph = np.zeros((n,n), dtype='float32')

    for i in range(0,n):
        copy = np.tile(np.array(x[i, :]), (i+1, 1))
        difference = copy - x[0:i+1, :]
        column = np.exp(-sigma*(difference**2).sum(1))

        #w_graph[0:i+1, i] = column
        w_graph[0:i, i] = column[:-1] # set diagonal to 0 the resulting graph is different

    return w_graph + w_graph.T


def get_data(path, sigma):
    """ Given a .csv features:label it returns the dataset modified with a gaussian kernel
    :param name: the path to the .csv
    :param sigma: sigma of the gaussian kernel
    :return: NG, GT, labels
    """
    df = pd.read_csv(path, delimiter=',', header=None)
    labels = df.iloc[:,-1].astype('category').cat.codes.values
    features = df.values[:,:-1].astype('float32')

    unq_labels, unq_counts = np.unique(labels, return_counts=True)

    NG = graph_from_points(features, sigma)
    aux, GT, aux2 = custom_cluster_matrix(len(labels), unq_counts, 0, 0)

    return NG.astype('float32'), GT.astype('int32'), labels


def get_GCoil1_data(path):
    data = sp.loadmat(path)
    NG = data['GCoil1']
    GT  = cluster_matrix(72, 20, 0, 0, 'constant', 0)
    return NG.astype('float32'), GT.astype('int32'), np.repeat(np.array(range(0,20)), 72)


def get_XPCA_data(path, sigma=0, to_remove=0):

    data = sp.loadmat(path)
    NG = graph_from_points(data['X'], sigma, to_remove)
    # Generates the custo GT wrt the number of rows removed
    tot_dim = 10000-to_remove
    c_dimensions = [1000]*int(tot_dim/1000)
    if tot_dim % 1000:
        c_dimensions.append(tot_dim % 1000)
    aux, GT, labels = custom_cluster_matrix(tot_dim, c_dimensions, 0, 0)

    return NG, GT, labels


def get_flicker32(path, sigma=0):
    data = sp.loadmat(path)
    #NG = graph_from_points(data['GISTall'], sigma, 0)
    NG = data['K25']
    GT  = cluster_matrix(70, 32, 0, 0, 'constant', 0)
    return NG.astype('float32'), GT.astype('int32'), np.repeat(np.array(range(0,32)), 70)

def synthetic_regular_partition(k, epsilon):
    """ Generates a synthetic regular partition.
    :param k: the cardinality of the partition
    :param epsilon: the epsilon parameter to calculate the number of irregular pairs
    :return: a weighted symmetric graph
    """

    # Generate a kxk matrix where each element is between (0,1]
    mat = np.tril(1-np.random.random((k, k)), -1)

    x = np.tril_indices_from(mat, -1)[0]
    y = np.tril_indices_from(mat, -1)[1]

    # Generate a random number between 0 and epsilon*k**2 (number of irregular pairs)
    n_irr_pairs = round(np.random.uniform(0, epsilon*(k**2)))

    # Select the indices of the irregular  pairs
    irr_pairs = np.random.choice(len(x), n_irr_pairs)

    mat[(x[irr_pairs],  y[irr_pairs])] = 0

    return mat + mat.T


def custom_cluster_matrix(mat_dim, dims, internoise_lvl, internoise_val, intranoise_lvl, intranoise_value):
    """ Custom noisy matrix
    :param mat_dim : int dimension of the whole graph
    :param dims: list(int) list of cluster dimensions
    :param internoise_lvl : float percentage of noise between clusters
    :param internoise_value : float value of the noise
    :param intranoise_lvl : float percentage of noise within clusters
    :param intranoise_value : float value of the noise

    :returns: np.array((n,n), dtype=float32) G the graph, np.array((n,n), dtype=int8) GT the ground truth, np.array() labels
    """
    if len(dims) > mat_dim:
        sys.exit("You want more cluster than nodes???")
        return 0

    if sum(dims) != mat_dim:
        sys.exit("The sum of clusters dimensions must be equal to the total number of nodes")
        return 0

    mat = np.tril(np.random.random((mat_dim, mat_dim)) < internoise_lvl, -1)
    mat = np.multiply(mat, internoise_val)

    GT = np.tril(np.zeros((mat_dim, mat_dim)), -1).astype('int8')

    x = 0
    for dim in dims:
        mat2 = np.tril(np.ones((dim,dim)), -1)

        if intranoise_value == 0:
            mat3 = np.tril(np.random.random((dim, dim)) < intranoise_lvl, -1)
            mat2 += mat3
            indices = (mat2 == 2)
            mat2[indices] = 0
            mat[x:x+dim,x:x+dim]= mat2
        else:
            mat3 = np.tril(np.random.random((dim, dim)) < intranoise_lvl, -1)
            mat3 = np.multiply(mat3, intranoise_value)
            mat2 += mat3
            indices = (mat2 > 1)
            mat2[indices] = intranoise_value
            mat[x:x+dim,x:x+dim]= mat2

        GT[x:x+dim,x:x+dim]= np.tril(np.ones(dim), -1).astype('int8')

        x += dim

    G = (mat + mat.T).astype('float32')
    return G, GT+GT.T, np.repeat(range(1, len(dims)+1,), dims)


def custom_crazy_cluster_matrix(mat_dim, dims, internoise_lvl, internoise_val, intranoise_lvl, intranoise_value):
    """ Custom noisy matrix
    :param mat_dim : dimension of the whole graph
    :param dims: list of cluster dimensions
    :param internoise_lvl : level of noise between clusters
    :param noise_lvl : value of the noise
    :returns: NG, GT, labels

    ---- 
    [TODO] works only with 4 clusters
    """
    if len(dims) > mat_dim:
        sys.exit("You want more cluster than nodes???")
        return 0

    if sum(dims) != mat_dim:
        sys.exit("The sum of clusters dimensions must be equal to the total number of nodes")
        return 0

    mat = np.tril(np.random.random((mat_dim, mat_dim)) < internoise_lvl, -1)
    mat = np.multiply(mat, internoise_val)
    GT = np.tril(np.zeros((mat_dim, mat_dim)), -1).astype('int8')
    x = 0
    i = 2
    for dim in dims:
        mat2 = np.tril(np.ones((dim,dim)), -1)

        if intranoise_value == 0:
            mat = mat.astype("float32")
            mat3 = np.tril(np.random.random((dim, dim)) < intranoise_lvl, -1)
            mat2 += mat3
            indices = (mat2 == 2)
            mat2[indices] = i/10
            mat[x:x+dim,x:x+dim]= mat2
        else:
            mat3 = np.tril(np.random.random((dim, dim)) < intranoise_lvl, -1)
            mat3 = np.multiply(mat3, intranoise_value)
            mat2 += mat3
            indices = (mat2 > 1)
            mat2[indices] = intranoise_value
            mat[x:x+dim,x:x+dim]= mat2

        GT[x:x+dim,x:x+dim]= np.tril(np.ones(dim), -1).astype('int8')

        x += dim
        i+=2
    m = (mat + mat.T).astype('float32')
    #print(np.unique(m.flatten()))
    return m, GT+GT.T, np.repeat(range(1, len(dims)+1,), dims)


