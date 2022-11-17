# packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import numba as nb


# functions
def kernel_euclidean(x, y):
    return np.linalg.norm(x - y)


def kernel_poly(x, y, r):
    return (1 + np.dot(x, y))**r


def kernel_gauss(x, y, gamma):
    return np.exp(-gamma * np.linalg.norm(x - y) ** 2)


def kmeans_euc(dat, K):
    # initialise parameters
    n, ncol = dat.shape
    best = 10 ** 12 + 1
    score = 10 ** 12
    # initialise clusters and distances of points to clusters
    C = np.clip(np.random.multivariate_normal(np.mean(dat, axis=0), np.diag(np.std(dat, axis=0)), size=(K,)), 0, 1)
    dists = np.zeros(shape=(n, K))
    # run until no improvement
    iter = 0
    while score < best:
        # update new best
        best = score
        # update distances
        for i in range(n):
            for k in range(K):
                dists[i, k] = kernel_euclidean(dat[i, :], C[k, :])
        # assign to clusters based on distances
        clusters = np.array([np.argmin(dists[i, :]) for i in range(n)])
        # update centroid locations
        for k in range(K):
            # ixgrid = np.ix_(np.where(clusters == k)[0], np.arange(ncol))
            # dat = data[ixgrid]
            # C[k, :] = np.sum(data[ixgrid], axis=0)
            C[k, :] = sum([dat[i, :] for i in range(n) if clusters[i] == k]) / np.count_nonzero(clusters == k)
        # calc score
        nclust = [np.count_nonzero(clusters == i) for i in range(K)]
        score = sum([sum([dists[i, k] for i in range(n) if clusters[i] == k]) for k in range(K)])
        # print division of clusters
        # print(iter)
        # print(nclust)
        # print(clusters)
        iter += 1
    return score


def kmeans_gauss(dat, K, gamma = 0.5):
    # initialise parameters
    n, ncol = dat.shape
    best = 10 ** 12 + 1
    score = 10 ** 12
    # initialise clusters and distances of points to clusters
    C = np.clip(np.random.multivariate_normal(np.mean(dat, axis=0), np.diag(np.std(dat, axis=0)), size=(K,)), 0, 1)
    #C = np.random.uniform(low=0, high=1, size=(K, ncol))
    dists = np.zeros(shape=(n, K))
    # run until no improvement
    iter = 0
    while score < best:
        # update new best
        best = score
        # update distances
        for i in range(n):
            for k in range(K):
                dists[i, k] = kernel_gauss(dat[i, :], C[k, :], gamma)
        # assign to clusters based on distances
        clusters = np.array([np.argmin(dists[i, :]) for i in range(n)])
        # update centroid locations
        #for k in range(K):
            # ixgrid = np.ix_(np.where(clusters == k)[0], np.arange(ncol))
            # dat = data[ixgrid]
            # C[k, :] = np.sum(data[ixgrid], axis=0)
            #C[k, :] = sum([dat[i, :] for i in range(n) if clusters[i] == k]) / np.count_nonzero(clusters == k)
        # calc score
        nclust = [np.count_nonzero(clusters == i) for i in range(K)]
        score = sum([sum([dists[i, k] for i in range(n) if clusters[i] == k]) for k in range(K)])
        # print division of clusters
        # print(iter)
        # print(nclust)
        # print(clusters)
        iter += 1
    return score


def kmeans_poly(dat, K, r):
    # initialise parameters
    n, ncol = dat.shape
    best = 10 ** 12 + 1
    score = 10 ** 12
    # initialise clusters and distances of points to clusters
    C = np.clip(np.random.multivariate_normal(np.mean(data, axis=0), np.diag(np.std(data, axis=0)), size=(K,)), 0, 1)
    #C = np.random.uniform(low=0, high=1, size=(K, ncol))
    dists = np.zeros(shape=(n, K))
    # run until no improvement
    iter = 0
    while score < best:
        # update new best
        best = score
        # update distances
        for i in range(n):
            for k in range(K):
                dists[i, k] = kernel_poly(dat[i, :], C[k, :], r)
        # assign to clusters based on distances
        clusters = np.array([np.argmin(dists[i, :]) for i in range(n)])
        # update centroid locations
        for k in range(K):
            # ixgrid = np.ix_(np.where(clusters == k)[0], np.arange(ncol))
            # dat = data[ixgrid]
            # C[k, :] = np.sum(data[ixgrid], axis=0)
            C[k, :] = sum([dat[i, :] for i in range(n) if clusters[i] == k]) / np.count_nonzero(clusters == k)
        # calc score
        nclust = [np.count_nonzero(clusters == i) for i in range(K)]
        score = sum([sum([dists[i, k] for i in range(n) if clusters[i] == k]) for k in range(K)])
        # print division of clusters
        # print(iter)
        # print(nclust)
        # print(clusters)
        iter += 1
    return score


def sim_matrix(X):
    # create empty similarity matrix
    W = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(i):
            W[i, j] = kernel_gauss(X[i, :],X[j, :], gam)
            W[j , i] = W[i, j]
    return W

@nb.njit(fastmath=True)
def jit_sim_matrix(X):
    W = np.zeros((X.shape[0], X.shape[0]))
    for i in nb.prange(X.shape[0]):
        for j in range(X.shape[0]):
            W[i, j] = kernel_gauss(X[i, :],X[j, :], gam)
    return W


def Laplacian(X):
    start = time.time()
    W = sim_matrix(X)
    print("similarity matrix time:",time.time()-start)
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    return L

def spectral_clustering(L,K, kernel):
    start = time.time()
    Lambdas, V = np.linalg.eig(L)
    print("Eigenvalue solver time:",time.time()-start)
    # Sort the eigenvalues by their L2 norms and record the indices
    ind = np.argsort(np.linalg.norm(np.reshape(Lambdas, (1, len(Lambdas))), axis=0))
    V_K = np.real(V[:, ind[:K]])
    return kmeans_dict[kernel](V_K,K)    


### MAIN
# read data
df = pd.read_csv('../LAOMLProject1/EastWestAirlinesCluster.csv')
df = df.drop(columns=['ID#'])
normalized_df = (df - df.min()) / (df.max() - df.min())
data = normalized_df.to_numpy()




# define parameters
nruns = 100
Ks = [1,2,3,4,5]
gam = 0.5
rr = 2
kernel_dict = {"euc":kernel_euclidean,"gauss":kernel_gauss, "poly": kernel_poly}
kmeans_dict = {"euc":kmeans_euc,"gauss":kmeans_gauss, "poly": kmeans_poly}

# Spectral Clustering
L = Laplacian(data)


for K in Ks:
    print("K = ", K)
    start = time.time()
    spec_clus = spectral_clustering(L, K, "gauss")
    time_clustering = time.time() - start
    print("Total spectral clustering time for K = ",K,": ", time_clustering)
    print("")
    print("")


