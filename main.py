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
    emptyclusters = True
    while emptyclusters:
        # initialise clusters and distances of points to clusters
        C = np.clip(np.random.multivariate_normal(np.mean(dat, axis=0), np.diag(np.std(dat, axis=0)), size=(K,)), 0, 1)
        # uncomment for uniform distribution of clusters instead
        # C = np.random.uniform(low=0, high=1, size=(K, ncol))
        dists = np.zeros(shape=(n, K))
        # update distances
        for i in range(n):
            for k in range(K):
                dists[i, k] = kernel_euclidean(dat[i, :], C[k, :])
        # assign to clusters based on distances
        clusters = np.array([np.argmin(dists[i, :]) for i in range(n)])
        if all([np.count_nonzero(clusters == k) > 0 for k in range(K)]):
            emptyclusters = False
        else:
            print("empty cluster, trying again")
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
            C[k, :] = sum([dat[i, :] for i in range(n) if clusters[i] == k]) / np.count_nonzero(clusters == k)
        # calc score
        nclust = [np.count_nonzero(clusters == i) for i in range(K)]
        score = sum([sum([dists[i, k]**2 for i in range(n) if clusters[i] == k]) for k in range(K)])
        iter += 1
    return score


def kmatval(kmat, dat, i, j, gamma):
    if kmat[i, j] != 0:
        return kmat[i, j]
    else:
        kmat[i, j] = kernel_gauss(dat[i, :], dat[j, :], gamma)
        kmat[j, i] = kmat[i, j]
        return kmat[i, j]

def kmeans_gauss(dat, K, gamma):
    # initialise parameters
    n, ncol = dat.shape
    best = 10 ** 12 + 1
    score = 10 ** 12
    kmat = np.zeros(shape=(n, n))
    emptyclusters = True
    while emptyclusters:
        # initialise clusters and distances of points to clusters
        #C = np.clip(np.random.multivariate_normal(np.mean(dat, axis=0), np.diag(np.std(dat, axis=0)), size=(K,)), 0, 1)
        # uncomment for uniform distribution of clusters instead
        C = np.random.uniform(low=0, high=1, size=(K, ncol))
        dists = np.zeros(shape=(n, K))
        # calc initial distances to clusters
        print("calcing init dists")
        for i in range(n):
            for k in range(K):
                dists[i, k] = kmatval(kmat, dat, i, i, gamma) - 2*kernel_gauss(dat[i, :], C[k, :], gamma) \
                              + kernel_gauss(C[k, :], C[k, :], gamma)
        # calc initial cluster assignment
        clusters = np.array([np.argmin(dists[i, :]) for i in range(n)])
        if all([np.count_nonzero(clusters == k) > 0 for k in range(K)]):
            emptyclusters = False
        else:
            print("empty cluster, trying again")
    # run until no improvement
    iter = 0
    while score < best:
        # update new best
        best = score
        # update distances
        print("updating dists")
        clustersums = [sum([kmatval(kmat, dat, i, j, gamma) for i in range(n) for j in range(n) \
                            if ((clusters[j] == k) and (clusters[i] == k))]) / np.count_nonzero(clusters == k)**2 \
                       for k in range(K)]
        for i in range(n):
            for k in range(K):
                dists[i, k] = kmatval(kmat, dat, i, i, gamma) - \
                              2*sum([kmatval(kmat, dat, i, j, gamma) for j in range(n) if (clusters[j] == k)]) \
                              / np.count_nonzero(clusters == k) + clustersums[k]
        # assign to clusters based on distances
        print("assigning clusts")
        clusters = np.array([np.argmin(dists[i, :]) for i in range(n)])
        # calc score
        score = sum([dists[i, k] for i in range(n) for k in range(K) if (clusters[i] == k)])
        iter += 1
    return score


def kmeans_gauss_2(dat, K, gamma):
    # initialise parameters
    n, ncol = dat.shape
    best = 10 ** 12 + 1
    score = 10 ** 12
    kmat = np.zeros(shape=(n, n))
    # fill kmat
    print("filling kmat")
    for i in range(n):
        kmat[i, i] = kernel_gauss(dat[i, :], dat[i, :], gamma)
        for j in range(i + 1, n):
            kmat[i, j] = kernel_gauss(dat[i, :], dat[j, :], gamma)
            kmat[j, i] = kmat[i, j]
    emptyclusters = True
    while emptyclusters:
        # initialise clusters and distances of points to clusters
        #C = np.clip(np.random.multivariate_normal(np.mean(dat, axis=0), np.diag(np.std(dat, axis=0)), size=(K,)), 0, 1)
        # uncomment for uniform distribution of clusters instead
        C = np.random.uniform(low=0, high=1, size=(K, ncol))
        dists = np.zeros(shape=(n, K))
        # calc initial distances to clusters
        print("calcing init dists")
        for k in range(K):
            clustersum = kernel_gauss(C[k, :], C[k, :], gamma)
            for i in range(n):
                dists[i, k] = kmat[i, i] - 2*kernel_gauss(dat[i, :], C[k, :], gamma) + clustersum
        # calc initial cluster assignment
        clusters = np.argmin(dists, axis=1)
        if all([np.count_nonzero(clusters == k) > 0 for k in range(K)]):
            emptyclusters = False
        else:
            print("empty cluster, trying again")
    # run until no improvement
    iter = 0
    while score < best:
        # update new best
        best = score
        # update distances
        print("updating dists")
        clustnumel = np.zeros(shape=(K,))
        clustersums2 = np.zeros(shape=(K,))
        for k in range(K):
            mask = np.where(clusters == k)
            clustnumel[k] = np.count_nonzero(clusters == k)
            clusterpartial = np.transpose(kmat[mask])
            clustersums2[k] = np.sum(clusterpartial[mask]) / (clustnumel[k] ** 2)
        # clustersums = [sum([kmatval(kmat, dat, i, j, gamma) for i in range(n) for j in range(n) \
        #                     if ((clusters[j] == k) and (clusters[i] == k))]) / np.count_nonzero(clusters == k)**2 \
        #                for k in range(K)]
        for k in range(K):
            mask = np.where(clusters == k)
            for i in range(n):
                dists[i, k] = kmat[i, i] - 2*np.sum(kmat[i, mask]) / clustnumel[k] + clustersums2[k]
        # assign to clusters based on distances
        print("assigning clusts")
        clusters = np.argmin(dists, axis=1)
        # calc score
        score = np.sum(np.choose(clusters, dists.transpose()))
        iter += 1
    return score


def kmatvalpoly(kmat, dat, i, j, r):
    if kmat[i, j] != 0:
        return kmat[i, j]
    else:
        kmat[i, j] = kernel_poly(dat[i, :], dat[j, :], r)
        kmat[j, i] = kmat[i, j]
        return kmat[i, j]


def kmeans_poly(dat, K, r):
    # initialise parameters
    n, ncol = dat.shape
    best = 10 ** 12 + 1
    score = 10 ** 12
    kmat = np.zeros(shape=(n, n))
    emptyclusters = True
    while emptyclusters:
        # initialise clusters and distances of points to clusters
        # C = np.clip(np.random.multivariate_normal(np.mean(dat, axis=0), np.diag(np.std(dat, axis=0)), size=(K,)), 0, 1)
        # uncomment for uniform distribution of clusters instead
        C = np.random.uniform(low=0, high=1, size=(K, ncol))
        dists = np.zeros(shape=(n, K))
        # calc initial distances to clusters
        print("calcing init dists")
        for i in range(n):
            for k in range(K):
                dists[i, k] = kmatvalpoly(kmat, dat, i, i, r) - 2 * kernel_poly(dat[i, :], C[k, :], r) \
                              + kernel_poly(C[k, :], C[k, :], r)
        # calc initial cluster assignment
        clusters = np.array([np.argmin(dists[i, :]) for i in range(n)])
        if all([np.count_nonzero(clusters == k) > 0 for k in range(K)]):
            emptyclusters = False
        else:
            print("empty cluster, trying again")
    # run until no improvement
    iter = 0
    while score < best:
        # update new best
        best = score
        # update distances
        print("updating dists")
        clustersums = [sum([kmatvalpoly(kmat, dat, i, j, r) for i in range(n) for j in range(n) \
                            if ((clusters[j] == k) and (clusters[i] == k))]) / np.count_nonzero(clusters == k) ** 2 \
                       for k in range(K)]
        for i in range(n):
            for k in range(K):
                dists[i, k] = kmatvalpoly(kmat, dat, i, i, r) - \
                              2*sum([kmatvalpoly(kmat, dat, i, j, r) for j in range(n) if (clusters[j] == k)]) \
                              / np.count_nonzero(clusters == k) + clustersums[k]
        # assign to clusters based on distances
        print("assigning clusts")
        clusters = np.array([np.argmin(dists[i, :]) for i in range(n)])
        # calc score
        nclust = [np.count_nonzero(clusters == i) for i in range(K)]
        score = sum([dists[i, k] for i in range(n) for k in range(K) if (clusters[i] == k)])
        iter += 1
    return score


def kmeans_poli2(dat, K, r):
    # initialise parameters
    n, ncol = dat.shape
    best = 10 ** 12 + 1
    score = 10 ** 12

    # initialise clusters and distances of points to clusters
    clusters = np.random.randint(0, K, size=n)
    dists = np.zeros(shape=(n, K))
    kernel_dists = np.zeros(shape=(n, n))
    kernel_dists_2 = np.zeros(shape=(K))

    for i in range(n):
        kernel_dists[i, i] = kernel_poly(dat[i, :], dat[i, :], r)
        for j in range(i + 1, n):
            kernel_dists[i, j] = kernel_poly(dat[i, :], dat[j, :], r)
            kernel_dists[j, i] = kernel_dists[i, j]

    # run until no improvement
    iter = 0

    while score < best:
        # update new best
        best = score
        # update distances
        for k in range(K):
            kernel_dists_2[k] = sum([kernel_dists[i, j] for i in range(n) for j in range(n) if
                                     ((clusters[i] == k) and (clusters[j] == k))]) / (
                                            np.count_nonzero(clusters == k) ** 2)

        for i in range(n):
            for k in range(K):
                dists[i, k] = kernel_dists[i, i] - 2 * sum(
                    [kernel_dists[i, j] for j in range(n) if clusters[j] == k]) / np.count_nonzero(clusters == k) + \
                              kernel_dists_2[k]
        # assign to clusters based on distances
        clusters = np.array([np.argmin(dists[i, :]) for i in range(n)])

        score = sum([sum([dists[i, k] for i in range(n) if clusters[i] == k]) for k in range(K)])
        # print division of clusters
        # print(iter)
        # print(nclust)
        # print(clusters)
        iter += 1

    return score

def sim_matrix(X):
    start = time.time()
    B = X.shape[0]
    W = np.zeros((B, B))
    for i in range(B):
        for j in range(i):
            W[i, j] = kernel_gauss(X[i, :], X[j, :],gam)
            W[j, i] = W[i, j]
    time_sim = time.time() - start
    print("Computation time similarity matrix: ",time_sim)
    return W

def Laplacian(X):
    W = sim_matrix(X)
    start = time.time()
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    time_lap = time.time() - start
    print("Computation time laplacian matrix: ",time_lap)
    return L

def normalized_Laplacian(X):
    W = sim_matrix(X)
    start = time.time()
    D_inv_half = np.diag(np.sqrt(np.sum(W, axis=1)))
    L_= np.identity(X.shape[0]) - D_inv_half*W*D_inv_half
    time_norm_lap = time.time() - start
    print("Computation time normalized laplacian matrix: ", time_norm_lap)
    return L_

def unnormalized_spectral_clustering(X,K):
    L = Laplacian(X)
    start = time.time()
    Lambdas, V = np.linalg.eig(L)
    time_eig = time.time() - start
    print("Computation time eigen-solver: ",time_eig)
    ind = np.argsort(np.linalg.norm(np.reshape(Lambdas, (1, len(Lambdas))), axis=0))
    V_K = np.real(V[:, ind[:K]])
    return kmeans_euc(V_K,K)

def normalized_spectral_clustering(X,K):
    L = normalized_Laplacian(X)
    start = time.time()
    Lambdas, V = np.linalg.eig(L)
    time_eig = time.time() - start
    print("Computation time eigen-solver: ", time_eig)
    ind = np.argsort(np.linalg.norm(np.reshape(Lambdas, (1, len(Lambdas))), axis=0))
    V_K = np.real(V[:, ind[:K]])
    return kmeans_euc(V_K, K)


### MAIN
# read data, normalize and store as numpy array
df = pd.read_csv('../LAOMLProject1/EastWestAirlinesCluster.csv')
df = df.drop(columns=['ID#'])
normalized_df = (df - df.min()) / (df.max() - df.min())
data = normalized_df.to_numpy()


# define parameters
nruns = 10
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
                score = kmeans_gauss_2(data, ki, gam)  # choose kernel method here
                timei = time.time() - start
            except ZeroDivisionError:
                print("division by zero somehow for for k="+str(ki)+",n="+str(ni)+", trying again; ")
            else:
                succes = True
                scores[ni, i] = score
                times[i] += timei
                print("  ==> score = " + str(score))
    print()
    times[i] = times[i] / nruns
    print("avg time per run is %.3f seconds" % times[i])


### PLOT
# achieved score versus number of clusters k
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
bp = ax.boxplot(scores)
ax.set_xticklabels([str(ki) for ki in Ks])
ax.set_xlabel('K')
ax.set_ylabel('Score')
plt.show()

# lineplot score vs k for kernel methods
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111)
# lp = ax.plot(Ks, scores.mean(axis=0))
# ax.set_xlabel('K')
# ax.set_ylabel('Score')
# plt.show()

# save avg time of kmeans per k to file
f = open('../LAOMLProject1/times gauss1 v2.txt', 'w')
f.write("k\td\n")
for i in range(nKs):
    f.write(f'{Ks[i]:10d}\t{times[i]:.3f}\n')
f.close()


