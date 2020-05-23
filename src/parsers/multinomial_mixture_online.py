import numpy as np
from random import sample
from copy import deepcopy
from scipy.optimize import root_scalar
from src.utils import get_vocabulary_indices
from global_utils import log_multi, multi, get_top_k_args
from src.parsers.base.log_parser_online import LogParserOnline
from global_constants import MAX_NEG_VALUE, CANNOT_LINK, ZERO_THRESHOLD, \
    MUST_LINK


class MultinomialMixtureOnline(LogParserOnline):
    def __init__(self, tokenized_logs, num_clusters, is_classification=True,
                 alpha=1, beta=1, epsilon=0.01, improvement_rate=1.25):
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.num_clusters = num_clusters
        self.is_classification = is_classification
        self.v_indices = get_vocabulary_indices(tokenized_logs)
        self.log_likelihood_history = []
        self.labeled_indices = []
        self.improvement_rate = improvement_rate

        self.t_c = None
        self.t_v = None
        self.t_c_obs = np.zeros((self.num_clusters, 1))
        self.t_v_obs = np.zeros((self.num_clusters, len(self.v_indices)))
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
        """
        log_labels: dictionary where each key is a true cluster and the values
                    are log indices.
        tokenized_logs: list of tokenized logs where the log_labels keys are a
                        subset.
        """
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
        """
        1) Randomly sample n_init logs
        Repeat the following steps "n_runs" times:
            2) Initialize parameters
            3) Run 1 EM
            4) Compute log-likelihood
        Keep parameter configuration and sufficient stats that result in the
        highest log-likelihood.
        """
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

    def enforce_constraints(self, constraints):
        """
        Enforce cannot-link constraints passed as a list of tuples where each
        tuple represents two logs that should not be clustered together.
        """
        self._enforce_must_link_constraints(constraints[MUST_LINK])
        self._enforce_cannot_link_constraints(constraints[CANNOT_LINK])

    def _get_empty_cluster(self):
        for g, v in enumerate(self.t_c):
            if abs(v - self.alpha + 1) < ZERO_THRESHOLD:
                return g
        return -1

    def _get_init_logs(self, logs, n_init):
        """
        Sample n_init logs without replacement.
        """
        init_indices = sample(range(len(logs)), k=n_init)
        return [logs[idx] for idx in init_indices]

    def _enforce_must_link_constraints(self, must_links):
        for link in must_links:
            log1, log2 = link
            c1 = self._get_token_counts(log1)
            c2 = self._get_token_counts(log2)

            r1 = self._get_responsibilities(c1)
            r2 = self._get_responsibilities(c2)

            g1_first, = get_top_k_args(r1, 1)
            g2_first, = get_top_k_args(r2, 1)

            if g1_first == g2_first:
                continue

            p1_first = r1[g1_first]
            p2_first = r2[g2_first]

            if p1_first < p2_first:
                self._change_dominant_resp(c1, g1_first, g2_first)
            else:
                self._change_dominant_resp(c2, g2_first, g1_first)

    def _enforce_cannot_link_constraints(self, cannot_links):
        for link in cannot_links:
            log1, log2 = link
            c1 = self._get_token_counts(log1)
            c2 = self._get_token_counts(log2)

            r1 = self._get_responsibilities(c1)
            r2 = self._get_responsibilities(c2)

            g1_first, g1_second = get_top_k_args(r1, 2)
            g2_first, g2_second = get_top_k_args(r2, 2)

            if g1_first != g2_first:
                continue

            p1_first = r1[g1_first]
            p2_first = r2[g2_first]

            g_empty = self._get_empty_cluster()
            if g_empty != -1 and p1_first < p2_first:
                self._change_dominant_resp(c1, g1_first, g_empty)
            elif g_empty != -1 and p1_first >= p2_first:
                self._change_dominant_resp(c2, g2_first, g_empty)
            elif p1_first < p2_first:
                self._change_dominant_resp(c1, g1_first, g1_second)
            else:
                self._change_dominant_resp(c2, g2_first, g2_second)

    def _init_sufficient_stats(self):
        self.t_c = self.t_c_obs + self.alpha - 1
        self.t_v = self.t_v_obs + self.beta - 1

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
            self.t_c_obs += r
            self.t_v_obs += r @ token_counts.T

        self.t_c += r
        self.t_v += r @ token_counts.T

    def _change_dominant_resp(self, c, g1, g):
        """
        Update the sufficient statistics so that cluster g dominates the current
        best cluster g1 by a factor based on the improvement_rate.

        After our update we want:
        improvement_rate = r[g] / r[g1]
        """
        t_xzyl_g1 = self.t_v[g1, :].reshape((-1, 1))
        t_xzyl_g = self.t_v[g, :].reshape((-1, 1))

        f = lambda a: float(np.log(self.t_c[g] + np.exp(a)) + np.sum(
            c * np.log(t_xzyl_g + c * np.exp(a))) - np.sum(
            c * np.log(np.sum(t_xzyl_g + c * np.exp(a)))) - np.log(
            self.t_c[g1]) - np.sum(c * np.log(t_xzyl_g1)) + np.sum(
            c * np.log(np.sum(t_xzyl_g1))) - np.log(self.improvement_rate))

        fprime = lambda a: float(
            np.exp(a) / (self.t_c[g] + np.exp(a)) + np.sum(
                (c ** 2) * np.exp(a) / (t_xzyl_g + c * np.exp(a))) - np.sum(
                c * np.sum(c) * np.exp(a)) / np.sum(t_xzyl_g + c * np.exp(a)))

        root = self._find_root(f, fprime)
        if root is not None:
            alpha = np.exp(root)
            self.t_c[g] += alpha
            self.t_v[g, :] += (alpha * c.flatten())
            self._update_parameters()

    def _find_root(self, f, fprime):
        for x0 in [2, 3, 5, 7, 11]:
            op = root_scalar(f=f, fprime=fprime, x0=x0, method='newton')
            if op.converged:
                return op.root
        return None

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
        self.pi = self.t_c / self.t_c.sum()
        self.theta = self.t_v / self.t_v.sum(axis=1)[:, np.newaxis]

    def _get_responsibilities(self, token_counts):
        log_multi_values = np.zeros((self.num_clusters, 1))
        for g in range(self.num_clusters):
            log_multi_values[g] = log_multi(token_counts, self.theta[g, :])
        log_multi_values -= np.max(log_multi_values)
        r = self.pi.reshape((-1, 1)) * np.exp(log_multi_values)
        return r / r.sum()
