import numpy as np
from copy import deepcopy
from random import sample
from src.parsers.base.log_parser import LogParser
from src.utils import get_vocabulary_indices, get_token_counts_batch
from global_utils import multi
from global_constants import MAX_NEG_VALUE


class MultinomialMixture(LogParser):
    def __init__(self, tokenized_logs, num_clusters,
                 verbose=False, epsilon=0.001):
        super().__init__(tokenized_logs)
        self.epsilon = epsilon
        self.num_clusters = num_clusters
        self.v_indices = get_vocabulary_indices(tokenized_logs)
        self.C = get_token_counts_batch(tokenized_logs, self.v_indices)
        self.Pi = np.zeros((num_clusters, 1))
        self.Theta = np.zeros((num_clusters, len(self.v_indices)))
        self.R = self._get_initial_responsibilities()
        self.labeled_indices = []
        self.verbose = verbose
        self.log_likelihood_history = []

    def parse(self, track_history=False):
        self._update_parameters()
        self._run_em_procedure(track_history)
        self._merge_clusters()

    def initialize_responsibilities(self, multinomial_mixture):
        self.R = deepcopy(multinomial_mixture.R)

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
                log = ' '.join(self.tokenized_logs[log_idx])
                print(log)
        print()

    def _get_initial_responsibilities(self):
        G = self.num_clusters
        D = len(self.tokenized_logs)
        R = np.zeros((G, D))
        for log_idx in range(D):
            R[np.random.randint(0, self.num_clusters), log_idx] = 1
        return R

    def _update_parameters(self):
        self.Pi = self.R.sum(axis=1, keepdims=True)
        self.Pi /= len(self.tokenized_logs)
        self.Theta = self.R @ self.C
        self.Theta /= self.Theta.sum(axis=1, keepdims=True)

    def _run_em_procedure(self, track_history):
        if track_history:
            self.log_likelihood_history.append(self._get_likelihood())
        current_ll, past_ll = None, None
        for _ in range(8):
            self._update_responsibilities()
            self._update_parameters()
            current_ll, past_ll = self._get_likelihood(), current_ll
            if self.verbose:
                print(current_ll)
            if track_history:
                self.log_likelihood_history.append(current_ll)
            if self._should_stop_em(current_ll, past_ll):
                break

    def _should_stop_em(self, current_ll, past_ll):
        if None in [current_ll, past_ll]:
            return False
        return abs((current_ll - past_ll) / past_ll) < self.epsilon

    def _update_responsibilities(self):
        G, D = self.R.shape
        for g in range(G):
            for d in self._get_non_labeled_doc_indices():
                self.R[g, d] = self.Pi[g] * self._get_multinomial_term(g, d)
        self.R /= self.R.sum(axis=0, keepdims=True)

    def _get_non_labeled_doc_indices(self):
        D = self.R.shape[1]
        return filter(lambda d: d not in self.labeled_indices, range(D))

    def _get_multinomial_term(self, g, d):
        return multi(self.C[d, :], self.Theta[g, :])

    def _get_likelihood(self):
        G, D = self.R.shape
        likelihood = 0
        for d in range(D):
            sum_term = 0
            for g in range(G):
                sum_term += self.Pi[g] * self._get_multinomial_term(g, d)
            if sum_term > 0:
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
