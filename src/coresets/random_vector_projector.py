import numpy as np
from math import sqrt
from global_utils import log_multi


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

    def get_projections(self):
        history = {}
        projections = []
        for n, x_n in enumerate(self.count_vectors):
            x_n_str = x_n.tostring()
            if x_n_str in history:
                projections.append(history[x_n_str])
            else:
                v_n = self._get_likelihood_projection(x_n)
                projections.append(v_n)
                history[x_n_str] = v_n
        return projections

    def _get_likelihood_projection(self, x_n):
        v_n = np.zeros((self.J, 1))
        for j in range(self.J):
            theta = self.param_samples[j]
            v_n[j] = self._get_log_likelihood(x_n, theta)
        return sqrt(1 / self.J) * v_n

    def _get_param_samples(self, cluster_pos, vocab_pos, J):
        return [self._sample_params(cluster_pos, vocab_pos) for _ in range(J)]

    def _get_dim_samples(self, D, J):
        return [np.random.randint(D) for _ in range(J)]

    def _sample_params(self, cluster_pos, vocab_pos):
        pi = np.random.dirichlet(cluster_pos.flatten())
        z = np.random.multinomial(1, pi)
        g = z.argmax()
        theta = np.random.dirichlet(vocab_pos[g])
        return theta

    def _get_log_likelihood(self, x_n, theta):
        return log_multi(x_n, theta)

    def _get_denominator(self, multi_values, pi):
        return (pi * multi_values).sum()
