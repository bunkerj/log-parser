import numpy as np
from math import sqrt

from global_utils import multi


class RandomVectorProjector:
    def __init__(self, count_vectors, cluster_posterior,
                 vocab_posterior, proj_vector_dim):
        self.num_clusters = cluster_posterior.size
        self.num_vocab = vocab_posterior.shape[1]
        self.count_vectors = count_vectors
        self.cluster_pos = cluster_posterior
        self.vocab_pos = vocab_posterior
        self.J = proj_vector_dim
        self.D = cluster_posterior.size + vocab_posterior.size
        self.param_samples = self._get_param_samples(self.num_clusters,
                                                     self.num_vocab, self.J)
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

    def _get_param_samples(self, num_clusters, num_vocab, J):
        return [self._sample_parameters(num_clusters, num_vocab) for _ in
                range(J)]

    def _get_dim_samples(self, D, J):
        return [np.random.randint(D) for _ in range(J)]

    def _sample_parameters(self, num_clusters, num_vocab):
        # TODO: add more specific posterior
        pi = np.random.dirichlet(np.ones(num_clusters))
        theta = np.random.dirichlet(np.ones(num_vocab),
                                    size=num_clusters)
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
