import functools
import numpy as np
from math import sqrt
from src.coresets.random_vector_projector import RandomVectorProjector
from src.utils import get_vocabulary_indices, get_token_counts


class GreedyIterativeGeodesicAscent:
    def __init__(self, tokenized_logs, cluster_pos=None, vocab_pos=None,
                 num_clusters=None):
        self._validate_arguments(cluster_pos, vocab_pos, num_clusters)
        self.tokenized_logs = tokenized_logs
        self.N = len(self.tokenized_logs)
        self.v_indices = get_vocabulary_indices(tokenized_logs)
        self.num_vocab = len(self.v_indices)
        self.num_clusters = self._get_num_clusters(cluster_pos, num_clusters)
        self.cluster_pos = self._get_cluster_pos(cluster_pos, self.num_clusters)
        self.vocab_pos = self._get_vocab_pos(vocab_pos, self.num_clusters,
                                             self.num_vocab)

    def get_coreset(self, subset_size, proj_dim):
        count_vectors = self._get_count_vectors()
        vector_projector = RandomVectorProjector(count_vectors,
                                                 self.cluster_pos,
                                                 self.vocab_pos,
                                                 proj_dim)
        projections = vector_projector.get_projections()
        weights = self.get_weights(projections, subset_size)
        reduced_weights = self._get_reduced_weights(weights)
        reduced_set = self._get_reduced_set(weights)
        reduced_indices = self._get_reduced_indices(weights)
        return reduced_weights, reduced_set, reduced_indices

    def get_weights(self, projections, subset_size):
        L = self._get_reference_vector(projections)
        l = L / self._norm(L)
        norm_projs = self._get_norm_projections(projections)
        w = np.zeros((self.N, 1))
        l_w = np.zeros(norm_projs[0].shape)

        for _ in range(subset_size):
            d_ref, dirs = self._get_geodesic_dirs(l, l_w, norm_projs)
            n = self._get_best_geodesic(d_ref, dirs)
            step_size = self._get_step_size(l, l_w, norm_projs, n)
            l_w, w = self._update_coreset(l_w, w, norm_projs, n, step_size)

        return self._scale_and_flatten_weights(projections, w, l_w, l, L)

    def _validate_arguments(self, cluster_pos, vocab_pos, num_clusters):
        """
        One of the following conditions must be met:
        1) Specified posterior distribution
        2) Specified number of clusters
        """
        is_valid_posterior = cluster_pos is not None and vocab_pos is not None
        is_valid_num_clusters = num_clusters is not None
        assert is_valid_posterior or is_valid_num_clusters

    def _get_num_clusters(self, cluster_pos, num_clusters):
        return cluster_pos.size if num_clusters is None else num_clusters

    def _get_cluster_pos(self, cluster_pos, num_clusters):
        return np.ones(num_clusters) if cluster_pos is None else cluster_pos

    def _get_vocab_pos(self, vocab_pos, num_clusters, num_vocab):
        return np.ones((num_clusters, num_vocab)) \
            if vocab_pos is None else vocab_pos

    def _get_count_vectors(self):
        return [get_token_counts(log, self.v_indices) for log in
                self.tokenized_logs]

    def _get_reference_vector(self, projections):
        return functools.reduce(lambda a, b: a + b, projections)

    def _get_reduced_weights(self, weights):
        return [w for w in weights if w > 0]

    def _get_reduced_set(self, weights):
        N = len(self.tokenized_logs)
        return [self.tokenized_logs[idx] for idx in range(N) if
                weights[idx] > 0]

    def _get_reduced_indices(self, weights):
        return [idx for idx, w in enumerate(weights) if w > 0]

    def _get_norm_projections(self, projections):
        return list(
            map(lambda v: v / self._norm(v) if self._norm(v) > 0 else v,
                projections))

    def _norm(self, v):
        return sqrt(v.T @ v)

    def _get_geodesic_dir(self, l_candidate, l_w):
        numerator = l_candidate - (l_candidate.T @ l_w) * l_w
        denominator = self._norm(numerator)
        return numerator / denominator if denominator > 0 \
            else np.zeros(l_w.shape)

    def _get_candidate_directions(self, l_w, norm_projs):
        return [self._get_geodesic_dir(l_n, l_w) for l_n in norm_projs]

    def _get_geodesic_dirs(self, l, l_w, norm_projections):
        d_ref = self._get_geodesic_dir(l, l_w)
        dirs = self._get_candidate_directions(l_w, norm_projections)
        return d_ref, dirs

    def _get_best_geodesic(self, d_ref, dirs):
        similarities = np.array([d_ref.T @ d_n for d_n in dirs])
        return similarities.argmax()

    def _get_step_size(self, l, l_w, norm_projections, n):
        l_n = norm_projections[n]
        c_0 = l.T @ l_n
        c_1 = l.T @ l_w
        c_2 = l_n.T @ l_w
        return (c_0 - c_1 * c_2) / ((c_0 - c_1 * c_2) + (c_1 - c_0 * c_2))

    def _update_coreset(self, l_w, w, norm_projs, n, step_size):
        l_n = norm_projs[n]

        l_w_num = (1 - step_size) * l_w + step_size * l_n
        l_w_denom = self._norm(l_w_num)
        l_w = l_w_num / l_w_denom

        one_n = self._get_basis(n)
        w_num = (1 - step_size) * w + step_size * one_n
        w = w_num / l_w_denom

        return l_w, w

    def _get_basis(self, n):
        one_n = np.zeros((self.N, 1))
        one_n[n] = 1
        return one_n

    def _scale_and_flatten_weights(self, projections, w, l_w, l, L):
        L_norm = self._norm(L)
        for n in range(self.N):
            L_n = projections[n]
            L_n_norm = self._norm(L_n)
            if L_n_norm > 0:
                w[n] *= float((L_norm / L_n_norm) * (l_w.T @ l))
            else:
                w[n] = 0
        return w.flatten()

    def _filter_projs(self, projs):
        return list(filter(lambda v_n: abs(self._norm(v_n)) > 0, projs))
