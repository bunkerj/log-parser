import numpy as np
from random import sample
from src.parsers.log_parser import LogParser

LIKELIHOOD_THRESHOLD = 1
ZERO_THRESHOLD = 0.0000001
MAX_NEG_VALUE = -99999999999


class MultinomialMixture(LogParser):
    def __init__(self, tokenized_log_entries, num_clusters, verbose=False):
        super().__init__(tokenized_log_entries)
        self.num_clusters = num_clusters
        self.v_indices = self._get_vocabulary_indices(tokenized_log_entries)
        self.C = self._get_counts(tokenized_log_entries)
        self.Pi = np.zeros((num_clusters, 1))
        self.Theta = np.zeros((num_clusters, len(self.v_indices)))
        self.R = self._get_initial_responsibilities()
        self.labeled_indices = []
        self.verbose = verbose

    def parse(self):
        self._update_parameters()
        self._run_em_procedure()
        self._merge_clusters()

    def initialize_responsibilities(self, multinomial_mixture):
        self.R = multinomial_mixture.R

    def label_logs(self, log_labels):
        for cluster_idx, log_indices in enumerate(log_labels.values()):
            for log_idx in log_indices:
                self.labeled_indices.append(log_idx)
                self.R[:, log_idx] = 0
                self.R[cluster_idx, log_idx] = 1

    def print_cluster_samples(self, n_samples):
        clusters = sorted(list(self.cluster_templates.keys()))
        for cluster_idx in clusters:
            print('\nCluster {}'.format(cluster_idx))
            n = min(len(self.cluster_templates[cluster_idx]), n_samples)
            log_indices = sample(self.cluster_templates[cluster_idx], n)
            for log_idx in log_indices:
                log_entry = ' '.join(self.tokenized_log_entries[log_idx])
                print(log_entry)
        print()

    def _get_counts(self, tokenized_log_entries):
        D = len(tokenized_log_entries)
        V = len(self.v_indices)
        C = np.zeros((D, V))
        for log_idx, tokenized_log_entry in enumerate(tokenized_log_entries):
            for token in tokenized_log_entry:
                if token in self.v_indices:
                    token_idx = self.v_indices[token]
                    C[log_idx, token_idx] += 1
        return C

    def _get_initial_responsibilities(self):
        G = self.num_clusters
        D = len(self.tokenized_log_entries)
        R = np.zeros((G, D))
        for log_idx in range(D):
            R[np.random.randint(0, self.num_clusters), log_idx] = 1
        return R

    def _update_parameters(self):
        self.Pi = self.R.sum(axis=1, keepdims=True)
        self.Pi /= len(self.tokenized_log_entries)
        self.Theta = self.R @ self.C + 1
        self.Theta /= self.Theta.sum(axis=1, keepdims=True)

    def _get_vocabulary_indices(self, tokenized_log_entries):
        v_indices = {}
        for tokens in tokenized_log_entries:
            for token in tokens:
                if token.isalpha():
                    v_indices[token] = 0
        for idx, token in enumerate(v_indices):
            v_indices[token] = idx
        return v_indices

    def _run_em_procedure(self):
        old_likelihood = None
        while True:
            self._update_responsibilities()
            self._update_parameters()
            likelihood = self._get_likelihood()
            if self.verbose:
                print(likelihood)
            if old_likelihood is not None and \
                    abs(old_likelihood - likelihood) < LIKELIHOOD_THRESHOLD:
                break
            old_likelihood = likelihood

    def _update_responsibilities(self):
        G, D = self.R.shape
        new_R = np.zeros(self.R.shape)
        for g in range(G):
            for d in range(D):
                new_R[g, d] = self.Pi[g] * self._get_multinomial_term(g, d)
        for d in self.labeled_indices:
            new_R[:, d] = self.R[:, d]
        self.R = new_R
        self.R /= self.R.sum(axis=0, keepdims=True)

    def _get_multinomial_term(self, k, d):
        return (self.Theta[k, :] ** self.C[d, :]).prod()

    def _get_likelihood(self):
        G, D = self.R.shape
        likelihood = 0
        for d in range(D):
            sum_term = 0
            for g in range(G):
                sum_term += self.Pi[g] * self._get_multinomial_term(g, d)
            if sum_term > ZERO_THRESHOLD:
                likelihood += np.log(sum_term)
            else:
                likelihood += MAX_NEG_VALUE
        return float(likelihood)

    def _merge_clusters(self):
        new_cluster_templates = {}
        D = self.R.shape[1]
        for d in range(D):
            cluster_idx = self.R[:, d].argmax()
            if cluster_idx not in new_cluster_templates:
                new_cluster_templates[cluster_idx] = []
            new_cluster_templates[cluster_idx].append(d)
        self.cluster_templates = new_cluster_templates
