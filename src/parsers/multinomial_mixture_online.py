import numpy as np
from random import sample
from copy import deepcopy
from global_constants import MAX_NEG_VALUE
from global_utils import multi
from src.parsers.base.log_parser_online import LogParserOnline
from src.utils import get_vocabulary_indices


class MultinomialMixtureOnline(LogParserOnline):
    def __init__(self, tokenized_logs, num_clusters,
                 is_classification=True, alpha=1, beta=1, epsilon=0.01):
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.num_clusters = num_clusters
        self.is_classification = is_classification
        self.v_indices = get_vocabulary_indices(tokenized_logs)
        self.log_likelihood_history = []
        self.labeled_indices = []

        self.t_zl = None
        self.t_xzyl = None
        self.t_l = np.zeros((self.num_clusters, 1))
        self.t_yl = np.zeros((self.num_clusters, len(self.v_indices)))
        self._init_sufficient_stats()

        self.pi = None
        self.theta = None
        self._init_parameters(self.num_clusters, len(self.v_indices))

    def perform_online_em(self, tokenized_log):
        self._update_sufficient_statistics(tokenized_log)
        self._update_parameters()

    def perform_online_batch_em(self, tokenized_logs, fixed_iter=1,
                                track_history=False):
        self._update_likelihood_hist(tokenized_logs, track_history)
        for _ in range(fixed_iter):
            for tokenized_log in tokenized_logs:
                self._update_sufficient_statistics(tokenized_log)
                self._update_parameters()
            self._update_likelihood_hist(tokenized_logs, track_history)

    def perform_offline_em(self, tokenized_logs, track_history=False):
        self._init_sufficient_stats()
        self._update_likelihood_hist(tokenized_logs, track_history)
        current_ll, past_ll = None, None
        while True:
            for tokenized_log in tokenized_logs:
                self._update_sufficient_statistics(tokenized_log)
            self._update_parameters()
            self._init_sufficient_stats()
            current_ll, past_ll = \
                self.get_log_likelihood(tokenized_logs), current_ll
            if track_history:
                self.log_likelihood_history.append(current_ll)
            if self._should_stop_offline_em(current_ll, past_ll):
                break

    def label_logs(self, log_labels, tokenized_logs):
        for cluster_idx, log_indices in enumerate(log_labels.values()):
            for log_idx in log_indices:
                self.labeled_indices.append(log_idx)
                tokenized_log = tokenized_logs[log_idx]
                self._update_sufficient_statistics(tokenized_log, cluster_idx)

    def get_log_likelihood(self, tokenized_logs):
        token_count_list = self._get_token_count_list(tokenized_logs)
        if self.is_classification:
            return self._get_classification_log_likelihood(token_count_list)
        else:
            return self._get_classical_log_likelihood(token_count_list)

    def get_log_likelihood_history(self):
        return self.log_likelihood_history

    def find_best_initialization(self, logs, n_init, n_runs=10):
        best_ll = None
        best_pi = None
        best_theta = None
        init_logs = self._get_init_logs(logs, n_init)

        for _ in range(n_runs):
            self._init_parameters(self.num_clusters, len(self.v_indices))
            for tokenized_log in init_logs:
                self._update_sufficient_statistics(tokenized_log)
            self._update_parameters()
            ll = self.get_log_likelihood(init_logs)
            if best_ll is None or ll > best_ll:
                best_ll = ll
                best_pi = self.pi
                best_theta = self.theta
            self._init_sufficient_stats()

        self.pi = best_pi
        self.theta = best_theta

        for tokenized_log in init_logs:
            self._update_sufficient_statistics(tokenized_log)

    def get_parameters(self):
        return deepcopy(self.pi), deepcopy(self.theta)

    def set_parameters(self, parameters):
        self.pi, self.theta = parameters

    def get_clusters(self, tokenized_logs):
        cluster_templates = {}
        for log_idx, tokenized_log in enumerate(tokenized_logs):
            token_counts = self._get_token_counts(tokenized_log)
            cluster_idx = self._get_best_cluster(token_counts)
            if cluster_idx not in cluster_templates:
                cluster_templates[cluster_idx] = []
            cluster_templates[cluster_idx].append(log_idx)
        return cluster_templates

    def _get_init_logs(self, logs, n_init):
        init_indices = sample(range(len(logs)), k=n_init)
        return [logs[idx] for idx in init_indices]

    def _init_sufficient_stats(self):
        self.t_zl = self.t_l + self.alpha - 1
        self.t_xzyl = self.t_yl + self.beta - 1

    def _update_likelihood_hist(self, tokenized_logs, track_history):
        if track_history:
            current_ll = self.get_log_likelihood(tokenized_logs)
            self.log_likelihood_history.append(current_ll)

    def _should_stop_offline_em(self, current_ll, past_ll):
        if None in [current_ll, past_ll]:
            return False
        return abs((current_ll - past_ll) / past_ll) < self.epsilon

    def _init_parameters(self, num_clusters, num_vocab):
        self.pi = np.random.dirichlet(np.ones(num_clusters))
        self.theta = np.random.dirichlet(np.ones(num_vocab),
                                         size=num_clusters)

    def _get_classification_log_likelihood(self, token_count_list):
        likelihood = 0
        for token_counts in token_count_list:
            g = self._get_best_cluster(token_counts)
            likelihood += np.log(
                self.pi[g] * multi(token_counts, self.theta[g, :]))
        return float(likelihood)

    def _get_classical_log_likelihood(self, token_count_list):
        likelihood = 0
        for token_counts in token_count_list:
            sum_term = 0
            for g in range(self.num_clusters):
                sum_term += self.pi[g] * multi(token_counts, self.theta[g, :])
            if sum_term > 0:
                likelihood += np.log(sum_term)
            else:
                likelihood += MAX_NEG_VALUE
        return float(likelihood)

    def _update_sufficient_statistics(self, tokenized_log, cluster_idx=-1):
        token_counts = self._get_token_counts(tokenized_log)

        if cluster_idx == -1:
            r = self._get_responsibilities(token_counts)
            if self.is_classification:
                g = int(r.argmax())
                r = np.zeros(r.shape)
                r[g] = 1
        else:
            r = np.zeros((self.num_clusters, 1))
            r[cluster_idx] = 1
            self.t_l += r
            self.t_yl += r @ token_counts.T

        self.t_zl += r
        self.t_xzyl += r @ token_counts.T

    def _get_best_cluster(self, token_counts):
        r = self._get_responsibilities(token_counts)
        return int(r.argmax())

    def _get_token_counts(self, tokenized_log):
        token_counts = np.zeros((len(self.v_indices), 1))
        for token in tokenized_log:
            if token in self.v_indices:
                token_counts[self.v_indices[token]] += 1
        return token_counts

    def _get_token_count_list(self, tokenized_logs):
        return [self._get_token_counts(tokenized_log) for tokenized_log in
                tokenized_logs]

    def _update_parameters(self):
        self.pi = self.t_zl / self.t_zl.sum()
        self.theta = self.t_xzyl / self.t_xzyl.sum(axis=1)[:, np.newaxis]

    def _get_responsibilities(self, token_counts):
        r = np.zeros((self.num_clusters, 1))
        for g in range(self.num_clusters):
            r[g] = self.pi[g] * multi(token_counts, self.theta[g, :])
        return r / r.sum()
