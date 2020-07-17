import numpy as np
from math import sqrt

from global_utils import multi


class RandomVectorProjector:
    def __init__(self, count_vectors, cluster_pos, vocab_pos, proj_vector_dim):
        self.num_clusters = cluster_pos.size
        self.num_vocab = vocab_pos.shape[1]
        self.count_vectors = count_vectors
        self.cluster_pos = cluster_pos
        self.vocab_pos = vocab_pos
        self.J = proj_vector_dim
        self.D = cluster_pos.size + vocab_pos.size
        self.param_samples = self._get_param_samples(self.cluster_pos,
                                                     self.vocab_pos,
                                                     self.J)
        self.dim_samples = self._get_dim_samples(self.D, self.J)

    def get_fw_projections(self):
        history = {}
        projections = []
        for n, x_n in enumerate(self.count_vectors):
            x_n_str = x_n.tostring()
            if x_n_str in history:
                projections.append(history[x_n_str])
            else:
                v_n = self._get_fw_projection(x_n)
                projections.append(v_n)
                history[x_n_str] = v_n

        return projections

    def _get_fw_projection(self, x_n):
        v_n = np.zeros((self.J, 1))
        for j in range(self.J):
            pi, theta = self.param_samples[j]
            d = self.dim_samples[j]
            v_n[j] = self._get_log_likelihood_derivative(x_n, pi, theta, d)
        return sqrt(self.D / self.J) * v_n

    def _get_param_samples(self, cluster_pos, vocab_pos, J):
        return [self._sample_parameters(cluster_pos, vocab_pos) for _ in
                range(J)]

    def _get_dim_samples(self, D, J):
        return [np.random.randint(D) for _ in range(J)]

    def _sample_parameters(self, cluster_pos, vocab_pos):
        pi = np.random.dirichlet(cluster_pos)
        theta = np.vstack([np.random.dirichlet(vp) for vp in vocab_pos])
        return pi, theta

    def _get_log_likelihood_derivative(self, x_n, pi, theta, d):
        multi_values = np.zeros((self.num_clusters, 1))
        for g in range(self.num_clusters):
            multi_values[g] = multi(x_n, theta[g, :])
        denom = (pi.reshape((-1, 1)) * multi_values.sum()).sum()
        if d < self.num_clusters:
            idx = d
            return multi_values[idx] / denom
        else:
            idx = d - self.num_clusters
            g = idx % self.num_clusters
            v = idx - (idx // self.num_clusters) * self.num_clusters
            return pi[g] * multi_values[g] * x_n[v] / (theta[g, v] * denom)
