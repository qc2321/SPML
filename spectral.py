import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import SpectralClustering

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


def compute_sc_struct(x, k=NUM_CLUSTERS):
    clustering = SpectralClustering(n_clusters=k,
                                    affinity='nearest_neighbors').fit(x)
    labels = clustering.labels_
    score = metrics.silhouette_score(x, labels)
    return labels, score


if __name__ == '__main__':
    rand_data, tight_clusters, wide_clusters, donut_rings = generate_data()

    sc_labels_rand, sc_struct_rand = compute_sc_struct(rand_data)
    sc_labels_tight, sc_struct_tight = compute_sc_struct(tight_clusters)
    sc_labels_wide, sc_struct_wide = compute_sc_struct(wide_clusters)
    sc_labels_donut, sc_struct_donut = compute_sc_struct(donut_rings)
    print(f'The Spectral Clustering struct loss for random data is {sc_struct_rand}.')
    print(f'The Spectral Clustering struct loss for tight clusters is {sc_struct_tight}.')
    print(f'The Spectral Clustering struct loss for wide clusters is {sc_struct_wide}.')
    print(f'The Spectral Clustering struct loss for donut rings is {sc_struct_donut}.\n')

    plt.scatter(donut_rings[:, 0], donut_rings[:, 1], c=sc_labels_donut, cmap='plasma')
    plt.show()
