import numpy as np
import matplotlib.pyplot as plt

NUM_CLUSTERS = 5
MAX_RANGE = 10000


def generate_data(n=2000, d=2, k=NUM_CLUSTERS, max_range=MAX_RANGE):
    # Random data
    x1 = np.random.rand(n, d)
    x1 = (x1 - 0.5) * 2 * max_range

    # Tight clusters
    x2 = np.zeros((n, d))
    x2_means = np.zeros((k, d))
    x2_min_dist = np.sqrt(d) / k
    for i in range(k):
        while True:
            x2_means[i] = np.random.rand(d)
            dist = [np.linalg.norm(x2_means[j] - x2_means[i])
                    for j in range(k) if j != i]
            if all(dist > x2_min_dist):
                break
    for i in range(k):
        means = (x2_means[i] - 0.5) * 2 * max_range
        cov = np.eye(d) * 10 * max_range
        start = round(i * n / k)
        end = start + round(n / k)
        x2[start:end, :] = np.random.multivariate_normal(means, cov, round(n / k))

    # Wide clusters
    x3 = np.zeros((n, d))
    x3_means = np.zeros((k, d))
    x3_min_dist = np.sqrt(d) / k
    for i in range(k):
        while True:
            x3_means[i] = np.random.rand(d)
            dist = [np.linalg.norm(x3_means[j] - x3_means[i])
                    for j in range(k) if j != i]
            if all(dist > x3_min_dist):
                break
    for i in range(k):
        means = (x3_means[i] - 0.5) * 2 * max_range
        cov = np.eye(d) * 100 * max_range
        start = round(i * n / k)
        end = start + round(n / k)
        x3[start:end, :] = np.random.multivariate_normal(means, cov, round(n / k))

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


def compute_dist_matrix(x):
    sq_term = np.sum(x**2, axis=1)
    left_term = sq_term.reshape(-1, 1)
    right_term = sq_term.reshape(1, -1)
    cross_term = x @ x.T
    dist_matrix = np.sqrt(left_term - 2 * cross_term + right_term)
    return dist_matrix


def struct_score(x, k=NUM_CLUSTERS):
    n = x.shape[0]
    dist_x = np.nan_to_num(compute_dist_matrix(x))
    mean_dist = np.sum(dist_x) / dist_x.size
    norm_dist_x = np.sqrt(dist_x / mean_dist)
    norm_dist = np.sum(norm_dist_x) / dist_x.size
    opt_dist = np.sqrt((k - 1) / k)
    sigma = (np.sqrt(k / (k-1)) - np.sqrt(n / (n-1))) / 3     # Adjust number of std dev
    score = np.exp(-(norm_dist - opt_dist)**2 / sigma**2)
    return score


if __name__ == '__main__':
    rand_data, tight_clusters, wide_clusters, donut_rings = generate_data()

    print(f'The struct score of random data is {struct_score(rand_data)}.')
    print(f'The struct score of tight clusters is {struct_score(tight_clusters)}.')
    print(f'The struct score of wide clusters is {struct_score(wide_clusters)}.')
    # print(f'The struct score of donut rings is {struct_score(donut_rings)}.')

    # plt.scatter(rand_data[:, 0], rand_data[:, 1])
    plt.scatter(tight_clusters[:, 0], tight_clusters[:, 1])
    # plt.scatter(wide_clusters[:, 0], wide_clusters[:, 1])
    # plt.scatter(donut_rings[:, 0], donut_rings[:, 1])
    plt.xlim([-MAX_RANGE, MAX_RANGE])
    plt.ylim([-MAX_RANGE, MAX_RANGE])
    plt.show()
