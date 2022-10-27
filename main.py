### INIT
# packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


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
    C = np.clip(np.random.multivariate_normal(np.mean(data, axis=0), np.diag(np.std(data, axis=0)), size=(K,)), 0, 1)
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


def kmeans_gauss(dat, K, gamma):
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
                dists[i, k] = kernel_gauss(dat[i, :], C[k, :], gamma)
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


### MAIN
# read data
df = pd.read_csv('../LAOMLProject1/EastWestAirlinesCluster.csv')
df = df.drop(columns=['ID#'])
normalized_df = (df - df.min()) / (df.max() - df.min())
data = normalized_df.to_numpy()


# define parameters
nruns = 100
Ks = [2, 3, 4, 5, 6, 7, 8]
nKs = len(Ks)
gam = 0.5
rr = 2


# run kmeans
scores = np.zeros(shape=(nruns, nKs))
times = np.zeros(shape=(nKs,))
for i in range(nKs):
    ki = Ks[i]
    print("k = "+str(ki)+" ...")
    for ni in range(nruns):
        succes = False
        while not succes:
            try:
                start = time.time()
                score = kmeans_gauss(data, ki, rr)
                timei = time.time() - start
            except ZeroDivisionError:
                print("empty cluster for k="+str(ki)+",n="+str(ni)+", trying again; ", end='')
            else:
                succes = True
                scores[ni, i] = score
                times[i] += timei
    print()
    times[i] = times[i] / nruns
    print("avg time per run is %.3f seconds" % times[i])


# plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
bp = ax.boxplot(scores)
ax.set_xticklabels([str(ki) for ki in Ks])
ax.set_xlabel('K')
ax.set_ylabel('Score')
plt.show()

# save times to file
f = open('../LAOMLProject1/times gauss05.txt')
for i in range(nKs):
    f.write("%d\t$.3f\n" % Ks[i], times[i])
f.close()
