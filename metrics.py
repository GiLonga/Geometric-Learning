import numpy as np
import scipy

def dunn_index(leaves):
    """
    Compute the Dunn index for a set of 2D curves.
    Parameters
    ----------
    leaves : np.array
        An array of shape (M, N, 2) where M is the number of curves,
        N is the number of points per curve.
    Returns
    -------
    dunn_index : float
        The Dunn index of the set of curves.
    Note
    ----   
    The curves are assumed to be grouped in 15 clusters of 50 curves each.
    """
    Nb,N,d = leaves.shape
    dist = np.zeros((Nb, Nb))
    for i in range(Nb):
        for j in range(Nb):
            dist[i,j] = np.linalg.norm(leaves[i,:,:]-leaves[j,:,:]) / np.sqrt(N)
    num_clusters = 15
    cluster_size = 50
    dk = np.zeros(num_clusters) # Intra-cluster distance
    ck = np.zeros((num_clusters, N ,d)) # Cluster centroids
    for k in range(num_clusters):
        start = k * cluster_size
        end = (k+1) * cluster_size
        dk[k] = np.max(dist[start:end, start:end])
        ck[k,:,:] = np.mean(leaves[start:end], axis=0)
    max_dk = np.max(dk)

    dk1k2 = np.zeros((num_clusters,num_clusters)) # Compute the distance between every pair of cluster centroids
    for k1 in range(num_clusters):
        for k2 in range(num_clusters):
            dk1k2[k1,k2] = np.linalg.norm(ck[k1,:,:]-ck[k2,:,:]) / np.sqrt(N)
    #max_dk1k2 = np.max(dk1k2)
    np.fill_diagonal(dk1k2, 2*np.max(dk1k2)) # Avoid zero distances on the diagonal
    min_dk1k2 = np.min(dk1k2)
    _dunn_index = min_dk1k2/max_dk
    return _dunn_index

if __name__ == "__main__":
    PATH = r"C:\Users\LONGA\Downloads\leaves_parameterized.mat"
    #path = #INSERT THE PATH TO YOUR DATA
    A = scipy.io.loadmat(PATH)
    print(dunn_index(A['leaves_parameterized']))
