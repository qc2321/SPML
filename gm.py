import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

NUM_CLUSTERS = 5


def generate_data(n=2000, d=2, k=NUM_CLUSTERS, max_range=10000):
    # Random data
    x1 = np.random.rand(n, d)
    x1 = (x1 - 0.5) * 2 * max_range

    # Tight clusters
    x2 = np.zeros((n, d))
    for i in range(k):
        means = np.random.rand(d)
        means = (means - 0.5) * 2 * max_range
        cov = np.eye(d) * 10 * max_range
        start = i * n // k
        end = start + n // k
        x2[start:end, :] = np.random.multivariate_normal(means, cov, n // k)

    # Wide clusters
    x3 = np.zeros((n, d))
    for i in range(k):
        means = np.random.rand(d)
        means = (means - 0.5) * 2 * max_range
        cov = np.eye(d) * 100 * max_range
        start = i * n // k
        end = start + n // k
        x3[start:end, :] = np.random.multivariate_normal(means, cov, n // k)

    # Donut rings
    x4 = np.zeros((n, d))
    i = 0
    while i < n:
        rand_pt = (np.random.rand(d) - 0.5) * 2 * max_range
        radius = np.sqrt(np.sum(np.square(rand_pt)))
        normalized_radius = radius % (max_range // k)
        if radius < max_range and normalized_radius > ((max_range // k) * 3 / 4):
            x4[i] = rand_pt
            i += 1

    return x1, x2, x3, x4


def compute_gm_struct(x, k=NUM_CLUSTERS):
    gm = GaussianMixture(n_components=k).fit(x)
    distance = means_dist(gm.means_)
    spread = covar_det(gm.covariances_)
    return np.log(distance / spread)


def compute_bgm_struct(x):
    bgm = BayesianGaussianMixture(n_components=int(np.log(x.shape[0])),
                                  n_init=int(np.log(x.shape[0])),
                                  weight_concentration_prior=1).fit(x)
    distance = means_dist(bgm.means_)
    spread = covar_det(bgm.covariances_)
    return np.log(distance / spread)


def means_dist(means):
    dist = 0
    for i in range(means.shape[0] - 1):
        for j in range(i+1, means.shape[0]):
            diff = means[i] - means[j]
            dist += np.linalg.norm(diff)
    return dist


def covar_det(covariances):
    det = 0
    for i in range(covariances.shape[0]):
        det += np.linalg.det(covariances[i])
    return det


if __name__ == '__main__':
    rand_data, tight_clusters, wide_clusters, donut_rings = generate_data()

    gm_struct_rand = compute_gm_struct(rand_data)
    gm_struct_tight = compute_gm_struct(tight_clusters)
    gm_struct_wide = compute_gm_struct(wide_clusters)
    gm_struct_donut = compute_gm_struct(donut_rings)
    print(f'The Gaussian Mixture struct loss for random data is {gm_struct_rand}.')
    print(f'The Gaussian Mixture struct loss for tight clusters is {gm_struct_tight}.')
    print(f'The Gaussian Mixture struct loss for wide clusters is {gm_struct_wide}.')
    print(f'The Gaussian Mixture struct loss for donut rings is {gm_struct_donut}.\n')

    bgm_struct_rand = compute_bgm_struct(rand_data)
    bgm_struct_tight = compute_bgm_struct(tight_clusters)
    bgm_struct_wide = compute_bgm_struct(wide_clusters)
    bgm_struct_donut = compute_bgm_struct(donut_rings)
    print(f'The Bayesian Gaussian Mixture struct loss for random data is {bgm_struct_rand}.')
    print(f'The Bayesian Gaussian Mixture struct loss for tight clusters is {bgm_struct_tight}.')
    print(f'The Bayesian Gaussian Mixture struct loss for wide clusters is {bgm_struct_wide}.')
    print(f'The Bayesian Gaussian Mixture struct loss for donut rings is {bgm_struct_donut}.')

    plt.scatter(rand_data[:, 0], rand_data[:, 1])
    plt.scatter(tight_clusters[:, 0], tight_clusters[:, 1])
    plt.scatter(wide_clusters[:, 0], wide_clusters[:, 1])
    plt.scatter(donut_rings[:, 0], donut_rings[:, 1])
    plt.show()
