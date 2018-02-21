import numpy as np
from sklearn import metrics
from scipy import stats


def l2(m1, m2):
    """ Computes the normalized L2 distance between two matrices of the same size
    :param m1: np.array
    :param m2: np.array
    :returns: float, l2 norm
    """
    #return np.sqrt(((m2-m1)**2).sum()) / m1.shape[0]
    return np.linalg.norm(m2-m1)/m2.shape[0]


def l1(m1, m2):
    """ Computes the normalized L1 distance between two matrices of the same size
    :param m1: np.array
    :param m2: np.array
    :returns: float, l2 norm
    """
    #return np.abs(m1 - m2).sum()/m2.shape[0]**2
    return np.linalg.norm(m2-m1, ord=1)/m2.shape[0]


def KL_divergence(d1, d2):
    """ Computes thhe kulback liebeler divergence between two vectors.
    It returns a tuple since the kl divergence is not symmetric
    :param m1: np.array()
    :param m1: np.array()
    :returns: np.array(float64) feature vector of measures
    """
    return stats.entropy(d1, d2), stats.entropy(d2, d1)


def ARI_KVS(m, labeling, ks=[5, 7, 9]):
    """ Implements knn voting system clustering, then compares with the correct labeling
    :param m: np.array((n,n))
    :param labeling: np.array(n) correct clustering
    :returns: adjusted random score
    """

    n = len(labeling)

    max_ars = -10

    for k in ks:
        candidates = np.zeros(n, dtype='uint32')
        i = 0
        for row in m:
            max_k_idxs = row.argsort()[-k:]
            aux = row[max_k_idxs] > 0
            k_indices = max_k_idxs[aux]

            if len(k_indices) == 0:
                k_indices = row.argsort()[-1:]

            #candidate_lbl = np.bincount(labeling[k_indices].astype(int)).argmax()
            candidate_lbl = np.bincount(labeling[k_indices]).argmax()
            candidates[i] = candidate_lbl
            i += 1

        ars = metrics.adjusted_rand_score(labeling, candidates)
        if ars > max_ars:
            max_k = k
            max_ars = ars

    return max_ars


def ARI_DS(m, labeling):
    """ Implements Dominant Set clustering, then compares with the correct labeling
    :param graph: reconstructed graph
    :returns: adjusted random score
    """
    clustering = dominant_sets(m)
    return metrics.adjusted_rand_score(clustering, labeling)


def replicator(A, x, inds, tol, max_iter):
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


def dominant_sets(graph_mat, max_k=4, tol=1e-5, max_iter=1000):
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

        y = replicator(graph_mat, x, np.where(~already_clustered)[0], tol, max_iter)
        cluster = np.where(y >= 1.0 / (graph_cardinality * 1.5))[0]
        already_clustered[cluster] = True
        clusters[cluster] = k
    clusters[~already_clustered] = k
    return clusters

