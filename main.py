### INIT
# packages
import pandas as pd
import numpy as np


# functions
def kernel_euclidean(x, y):
    return np.linalg.norm(x - y)


def kernel_poly(x, y, r):
    return (1 + np.dot(x, y)) ^ r


def kernel_gauss(x, y, gamma):
    return np.exp(-gamma * np.linalg.norm(x - y) ** 2)


# read data
df = pd.read_csv('../LAOMLProject1/EastWestAirlinesCluster.csv')
df = df.drop(columns=['ID#'])
normalized_df = (df - df.min()) / (df.max() - df.min())
data = normalized_df.to_numpy()


# define parameters
K = 4
ncol = 11
n = 3998



### MAIN
# initialise clusters
#C = np.random.uniform(low=0, high=1, size=(K, ncol))
C = np.clip(np.random.multivariate_normal(np.mean(data, axis=0), np.diag(np.std(data, axis=0)), size=(K,)), 0, 1)
dists = np.zeros(shape=(n, K))


# run kmeans
best = 10**12 + 1
score = 10**12
iter = 0
while score < best:
    # update new best
    best = score
    # update distances
    for i in range(n):
        for k in range(K):
            dists[i, k] = kernel_euclidean(data[i, :], C[k, :])
    # assign to clusters based on distances
    clusters = np.array([np.argmin(dists[i, :]) for i in range(n)])
    # update centroid locations
    for k in range(K):
        #ixgrid = np.ix_(np.where(clusters == k)[0], np.arange(ncol))
        #dat = data[ixgrid]
        #C[k, :] = np.sum(data[ixgrid], axis=0)
        C[k, :] = sum([data[i, :] for i in range(n) if clusters[i] == k]) / np.count_nonzero(clusters == k)
    # calc score
    nclust = [np.count_nonzero(clusters == i) for i in range(K)]
    score = sum([sum([dists[i, k] for i in range(n) if clusters[i] == k]) for k in range(K)])
    # print division of clusters
    print(iter)
    print(nclust)
    print(clusters)
    iter += 1


# ones = np.ones(shape=(n,))
# kernelmat = np.zeros(shape=(n, n))
# for i in range(n):
#     for j in range(i, n):
#         kernelmat[i, j] = kernel_euclidean(data[i, :], data[j, :])
#         kernelmat[j, i] = kernelmat[i, j]
#

# #while clusters_new is not clusters
# for z in range(2):
#     # update distance matrix
#     for k in range(K):
#         sizeCk = np.count_nonzero(clusters == k)
#         clustkerns = np.ix_(np.where(clusters == k), np.where(clusters == k))
#         totsumclust = 1/sizeCk**2 * sum(kernelmat[clustkerns])
#         for i in range(n):
#             dists[i, k] = kernel_euclidean(data[i, :], data[i, :]) - 2/sizeCk * sum((kernelmat[i, j] for j in range(n)
#                                                                                      if clusters[j] == k)) + totsumclust
#     # update best cluster
#     clusters = [np.argmin(dists[i, :]) for i in range(0, n)]


# test
print("\nFINAL RESULT:")
print("\nitems per cluster " + str(nclust))
print("\nscore " + str(score))

