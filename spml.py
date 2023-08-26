import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

INPUT_SIZE = 180
DIMENSIONS = 2
CLASSES = 2
SUBCLUSTERS = 2
MAX_RANGE = 100


def generate_data(n=INPUT_SIZE, d=DIMENSIONS, k=CLASSES, s=SUBCLUSTERS, max_range=MAX_RANGE):
    X = np.zeros((n, d))
    y = np.zeros(n)
    X_means = np.zeros((k * s, d))
    X_min_dist = np.sqrt(d / (k * s)) / 2       # Control spacing between clusters

    # Comment out all but one cluster center generation
    # 1) Random cluster centers
    # for i in range(k * s):
    #     while True:
    #         X_means[i] = np.random.rand(d)
    #         dist = [np.linalg.norm(X_means[j] - X_means[i])
    #                 for j in range(k) if j != i]
    #         if all(dist > X_min_dist):
    #             break

    # 2) Manual generation with d=2, k=2, s=2
    X_means[0] = np.array([0.1, 0.9])
    X_means[1] = np.array([0.9, 0.9])
    X_means[2] = np.array([0.1, 0.1])
    X_means[3] = np.array([0.9, 0.1])

    for i in range(k * s):
        means = (X_means[i] - 0.5) * 2 * max_range * 0.9    # Keep spread data within edges
        cov = np.eye(d) * max_range**2 / 300      # Control tightness of clusters
        start = round(i * n / (k * s))
        end = start + round(n / (k * s))
        # np.random.seed(42)
        X[start:end, :] = np.random.multivariate_normal(means, cov, round(n / (k * s)))
        y[start:end] = i % k
    return X, y


def transform_data(A_sqrt, X, n=INPUT_SIZE, d=DIMENSIONS):
    X_new = np.zeros((n, d))
    for i in range(n):
        X_new[i] = A_sqrt @ X[i]
    return X_new


class SPML:
    def __init__(self, X, y, tol=10, k=CLASSES, s=SUBCLUSTERS, max_range=MAX_RANGE):
        self.X = X
        self.y = y
        self.tol = tol
        self.n, self.d = X.shape
        self.S, self.D = self.group_similar(X, y)
        self.reg_S = 1 / self.S.shape[0]                    # Normalize sum of distances
        self.reg_D = 1 / self.D.shape[0]                    # Normalize sum of distances
        self.reg_struct = self.d**2 * k * s * max_range     # Struct hyperparam scales with d, k, s

    def group_similar(self, X, y):
        S = []
        D = []
        for i in range(self.n):
            for j in range(i+1, self.n):
                if y[i] == y[j]:
                    S.append((X[i], X[j]))
                else:
                    D.append((X[i], X[j]))
        return np.array(S), np.array(D)

    def fit(self, lr=1e-6, max_iter=1000):
        A = np.eye(self.d)       # initialize as identity matrix
        num_iter = 0
        cur_loss = self.calc_loss(A)
        while True:
            prev_loss = cur_loss
            dA = self.calc_dA(A)
            A -= lr * dA
            A = self.project_PSD(A)
            cur_loss = self.calc_loss(A)
            loss_diff = prev_loss - cur_loss
            num_iter += 1
            print(f'Iter #{num_iter}: prev loss, cur loss = {prev_loss}, {cur_loss}; loss diff = {loss_diff}')
            if loss_diff < self.tol or cur_loss < 50 or num_iter >= max_iter:
                break
        return np.real(sqrtm(A))

    def calc_loss(self, A, k=CLASSES):
        loss_S, loss_D, loss_struct = 0, 0, 0

        for i in range(self.S.shape[0]):
            S_ij = (self.S[i, 0] - self.S[i, 1]).reshape(self.d, 1)
            sq_dist = S_ij.T @ A @ S_ij
            loss_S += sq_dist.item()
        loss_S *= self.reg_S

        for i in range(self.D.shape[0]):
            D_ij = (self.D[i, 0] - self.D[i, 1]).reshape(self.d, 1)
            sq_dist = D_ij.T @ A @ D_ij
            loss_D += np.sqrt(sq_dist.item())
        loss_D *= self.reg_D

        for i in range(k):
            X_c = self.X[self.y == i]
            loss_struct += self.calc_struct_score(A, X_c)
        loss_struct *= self.reg_struct

        return loss_S - loss_D - loss_struct

    def calc_struct_score(self, A, X_c, s=SUBCLUSTERS):
        X_c_trans = X_c @ np.real(sqrtm(A))
        n = X_c_trans.shape[0]
        dist_mat_diff = self.calc_dist_mat_diff(X_c_trans, s)
        sigma = (np.sqrt(s / (s - 1)) - np.sqrt(n / (n - 1))) / 3   # Adjust number of std dev
        score = np.exp(-dist_mat_diff ** 2 / sigma ** 2)
        return score

    def calc_dist_mat_diff(self, X_c_trans, s):
        n = X_c_trans.shape[0]
        dist_mat = np.nan_to_num(self.calc_dist_matrix(X_c_trans))
        mean_dist = np.sum(dist_mat) / dist_mat.size
        norm_dist_mat = np.sqrt(dist_mat / mean_dist)
        norm_dist = np.sum(norm_dist_mat) / dist_mat.size
        opt_dist = n * (s - 1) / s * np.sqrt(s / (s - 1)) / (n - 1)
        return norm_dist - opt_dist

    def calc_dist_matrix(self, x):
        sq_term = np.sum(x ** 2, axis=1)
        left_term = sq_term.reshape(-1, 1)
        right_term = sq_term.reshape(1, -1)
        cross_term = x @ x.T
        sq_dist = np.abs(left_term - 2 * cross_term + right_term)
        dist_matrix = np.sqrt(sq_dist)
        return dist_matrix

    def calc_dA(self, A):
        dA_S = np.zeros((self.d, self.d))
        dA_D = np.zeros((self.d, self.d))
        dA_struct = np.zeros((self.d, self.d))

        for i in range(self.S.shape[0]):
            S_ij = (self.S[i, 0] - self.S[i, 1]).reshape(self.d, 1)
            dA_S += S_ij @ S_ij.T
        dA_S *= self.reg_S

        for i in range(self.D.shape[0]):
            D_ij = (self.D[i, 0] - self.D[i, 1]).reshape(self.d, 1)
            dA_D += D_ij @ D_ij.T / (2 * np.sqrt(D_ij.T @ A @ D_ij))
        dA_D *= self.reg_D

        for i in range(CLASSES):
            X_c = self.X[self.y == i]
            dA_struct += self.calc_dA_struct(A, X_c)
        dA_struct *= self.reg_struct

        return dA_S - dA_D - dA_struct

    def calc_dA_struct(self, A, X_c, s=SUBCLUSTERS):
        X_c_trans = X_c @ sqrtm(A)
        n = X_c_trans.shape[0]
        p = self.calc_struct_score(A, X_c, s)
        q = -2 * self.calc_dist_mat_diff(X_c_trans, s)

        # Decomposing quotient rule
        # v du/dx
        du = np.zeros((self.d, self.d))
        for i in range(n):
            for j in range(i+1, n):
                C_ij = (X_c[i] - X_c[j]).reshape(self.d, 1)
                du += C_ij @ C_ij.T / ((4 * C_ij.T @ A @ C_ij) ** 0.75) * 2
        v = 0
        for i in range(n):
            for j in range(i+1, n):
                C_ij = (X_c[i] - X_c[j]).reshape(self.d, 1)
                v += np.sqrt(C_ij.T @ A @ C_ij) * 2
        v = np.sqrt(v)
        v_du = du * v

        # u dv/dx
        u = 0
        for i in range(n):
            for j in range(i+1, n):
                C_ij = (X_c[i] - X_c[j]).reshape(self.d, 1)
                u += (C_ij.T @ A @ C_ij) ** 0.25 * 2
        dv = np.zeros((self.d, self.d))
        for i in range(n):
            for j in range(i+1, n):
                C_ij = (X_c[i] - X_c[j]).reshape(self.d, 1)
                dv += C_ij @ C_ij.T / (2 * np.sqrt(C_ij.T @ A @ C_ij)) * 2
        dv /= 2 * v
        u_dv = u * dv

        # v^2
        v_2 = v ** 2
        r = 1 / n * (v_du - u_dv) / v_2
        return p * q * r

    def project_PSD(self, A):
        eigvals, P = np.linalg.eig(A)
        D = np.diag(np.maximum(eigvals, 0))
        return P @ D @ np.linalg.inv(P)


def plot_graph(X, y, X_new):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(X[:, 0], X[:, 1], s=10, c=y, cmap='plasma')
    ax1.set_xlim([-MAX_RANGE, MAX_RANGE])
    ax1.set_ylim([-MAX_RANGE, MAX_RANGE])
    ax2.scatter(X_new[:, 0], X_new[:, 1], s=10, c=y, cmap='plasma')
    ax2.set_xlim([-MAX_RANGE, MAX_RANGE])
    ax2.set_ylim([-MAX_RANGE, MAX_RANGE])
    plt.show()


if __name__ == '__main__':
    train_X, train_y = generate_data()
    model = SPML(train_X, train_y)
    trans_mat = model.fit()
    X_trans = transform_data(trans_mat, train_X)
    if DIMENSIONS == 2:
        plot_graph(train_X, train_y, X_trans)
