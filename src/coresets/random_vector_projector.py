import numpy as np
from math import sqrt

from global_utils import multi


class RandomVectorProjector:
    def __init__(self, num_clusters, num_vocab, log_vectors, cluster_posterior,
                 vocab_posterior, proj_vector_dim):
        self.num_clusters = num_clusters
        self.num_vocab = num_vocab
        self.log_vectors = log_vectors
        self.cluster_pos = cluster_posterior
        self.vocab_pos = vocab_posterior
        self.J = proj_vector_dim
        self.D = num_clusters * (1 + num_vocab)

    def get_fw_projections(self):
        parameter_samples = [self._sample_parameters() for _ in
                             range(self.J)]
        dimension_samples = [np.random.randint(self.D) for _ in
                             range(self.J)]

        projections = []
        for n, x_n in enumerate(self.log_vectors):
            v_n = np.zeros((self.J, 1))
            for j in range(self.J):
                pi, theta = parameter_samples[j]
                d = dimension_samples[j]
                v_n[j] = self._get_log_likelihood_derivative(x_n, pi, theta, d)
            projections.append(sqrt(self.D / self.J) * v_n)

        return projections

    def _sample_parameters(self):
        # TODO: add more specific posterior
        pi = np.random.dirichlet(np.ones(self.num_clusters))
        theta = np.random.dirichlet(np.ones(self.num_vocab),
                                    size=self.num_clusters)
        return pi, theta

    def _get_log_likelihood(self, x_n, pi, theta):
        pass

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
