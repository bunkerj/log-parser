import numpy as np
from src.parsers.base.log_parser_online import LogParserOnline
from src.utils import get_vocabulary_indices
from scipy.stats import multinomial as multi


class MultinomialMixtureOnline(LogParserOnline):
    def __init__(self, num_clusters, tokenized_log_entries,
                 is_classification=True, alpha=2, beta=2):
        self.N = 0
        self.alpha = alpha
        self.beta = beta
        self.num_clusters = num_clusters
        self.is_classification = is_classification

        self.v_indices = get_vocabulary_indices(tokenized_log_entries)
        self.T_z = np.ones((num_clusters, 1)) / num_clusters
        self.T_xz = np.ones((num_clusters, len(self.v_indices))) \
                    / len(self.v_indices)

        self.Pi = np.random.dirichlet(np.ones(num_clusters))
        self.Theta = np.random.dirichlet(np.ones(len(self.v_indices)),
                                         size=num_clusters)

    def perform_online_em(self, tokenized_log):
        self._update_sufficient_statistics(tokenized_log)
        self._update_parameters()

    def perform_online_batch_em(self, tokenized_log_entries):
        for tokenized_log in tokenized_log_entries:
            self._update_sufficient_statistics(tokenized_log)
            self._update_parameters()

    def perform_offline_em(self, tokenized_log_entries, n_iter=1):
        for _ in range(n_iter):
            for tokenized_log in tokenized_log_entries:
                self._update_sufficient_statistics(tokenized_log)
            self._update_parameters()
            self.N = 0

    def get_classification_likelihood(self, tokenized_log_entries):
        cl = 0
        for tokenized_log in tokenized_log_entries:
            token_counts = self._get_token_counts(tokenized_log)
            n = sum(token_counts)
            g = self._get_responsibilities(token_counts)
            cl += np.log(
                self.Pi[g] * multi.pmf(token_counts, n, self.Theta[g, :]))
        return cl

    def get_clusters(self, tokenized_log_entries):
        cluster_templates = {}
        for log_idx, tokenized_log in enumerate(tokenized_log_entries):
            token_counts = self._get_token_counts(tokenized_log)
            cluster_idx = self._get_best_cluster(token_counts)
            if cluster_idx not in cluster_templates:
                cluster_templates[cluster_idx] = []
            cluster_templates[cluster_idx].append(log_idx)
        return cluster_templates

    def _update_sufficient_statistics(self, tokenized_log):
        token_counts = self._get_token_counts(tokenized_log)
        r = self._get_responsibilities(token_counts)

        if self.is_classification:
            cluster_idx = int(r.argmax())
            r = np.zeros(r.shape)
            r[cluster_idx] = 1

        curr_T_z = r + (self.alpha - 1)
        curr_T_xz = r @ token_counts.T + (self.beta - 1)

        if self.N == 0:
            self.T_z = curr_T_z
            self.T_xz = curr_T_xz
        else:
            self.T_z += (curr_T_z - self.T_z) / (self.N + 1)
            self.T_xz += + (curr_T_xz - self.T_xz) / (self.N + 1)

        self.N += 1

    def _get_best_cluster(self, token_counts):
        r = self._get_responsibilities(token_counts)
        return int(r.argmax())

    def _get_token_counts(self, tokenized_log):
        token_counts = np.zeros((len(self.v_indices), 1))
        for token in tokenized_log:
            if token in self.v_indices:
                token_counts[self.v_indices[token]] += 1
        return token_counts

    def _update_parameters(self):
        self.Pi = self.T_z / self.T_z.max()
        self.Theta = self.T_xz / self.T_xz.sum(axis=1)[:, np.newaxis]

    def _get_responsibilities(self, token_counts):
        r = np.zeros((self.num_clusters, 1))
        for g in range(self.num_clusters):
            r[g] = self.Pi[g] * self._unnormalized_multi(token_counts, g)
        return r / r.sum()

    def _unnormalized_multi(self, token_counts, g):
        return (self.Theta[g, :] ** token_counts.flatten()).prod()
