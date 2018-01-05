import numpy as np
import os
import sys
sys.path.append("../../")
import conf


def synthetic_graph(n, d):
    """ Generate a n,n graph with a given density d
    :param d: float density of the graph
    :return: np.array((n, n), dtype='float32') graph with density d
    """
    G = np.tril(np.random.random((n, n)) <= d, -1).astype('int8')
    return G + G.T


def synthethic_cheese_graph(n, d):
    """ Generate a n,n cheese-style (with holes) graph of a given density d
    :param d: float density of the graph
    :return: np.array((n, n), dtype='float32') graph with density d
    """

    l = [x for x in range(0,n+1)]

    groups = zip(l[::5], l[1::5], l[2::5])#, l[3::5]), l[4::5])
    groups = list(groups)

    a = np.array(groups[::2]).flatten()
    b = np.array(groups[1::2]).flatten()

    mat = np.ones((n,n))

    mat[np.ix_(a, b)] = 0
    mat += mat.T
    mat[np.where(mat == 1)] = 0
    mat /= 2

    np.fill_diagonal(mat, 0)

    #print(mat)
    #plt.imshow(mat)
    #plt.show()


def density(G):
    """ Check density of a synthetic graph
    :param G: np.array((n, n), dtype='int8') graph
    :return: float density of the graph
    """
    n = G.shape[0]
    e = np.where(G == 1)[0].size / 2
    return e / ((n*(n-1))/2)

######################################
######################################

# Params
n = conf.gen.n
DSET_PATH = conf.gen.DSET_PATH
densities = conf.gen.densities
n_graphs = conf.gen.n_graphs
tolerance = conf.gen.tolerance


if not os.path.exists(DSET_PATH):
    os.makedirs(DSET_PATH)

# Generation
for dens in densities:
    for dset_id in range(1, n_graphs+1):
        G = synthetic_graph(n, dens)
        assert G.dtype == 'int8', "Dtype is not correct"
        d = round(density(G), 2)
        assert dens-tolerance < d and d < dens+tolerance, f"Density: {dens} != {d}"
        filename = f"{n}_{dens:.2f}_{dset_id}"
        np.savez_compressed(DSET_PATH+filename, G=G, bounds=np.array([], dtype='float32'))
        print(f"[ OK ] {filename}")

print("Graphs generation completed")

