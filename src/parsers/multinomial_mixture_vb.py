import numpy as np
from collections import defaultdict
from scipy.special import digamma

from global_utils import log_multi
from src.parsers.base.log_parser import LogParser
from src.utils import get_token_counts_batch, get_vocabulary_indices


class MultinomialMixtureVB(LogParser):
    def __init__(self, tokenized_logs, num_clusters, epsilon=0.01):
        super().__init__(tokenized_logs)
        self.epsilon = epsilon
        self.v_indices = get_vocabulary_indices(tokenized_logs)
        self.C = get_token_counts_batch(tokenized_logs, self.v_indices)
        self.G = num_clusters
        self.N = len(tokenized_logs)
        self.V = len(self.v_indices)
        self.R = np.zeros((self.N, self.G))
        self.pi_v = np.zeros(self.G)
        self.theta_v = np.zeros((self.G, self.V))
        self.ex_ln_pi = np.zeros(self.G)
        self.ex_ln_theta = np.zeros((self.G, self.V))
        self.cluster_templates = {}
        self.alpha_0 = 1 / self.G
        self.beta_0 = 1 / self.V
        self.alpha = self.alpha_0 + np.zeros(self.G)
        self.beta = self.beta_0 + np.zeros((self.G, self.V))
        self.labeled_indices = []
        self.W = defaultdict(dict)
        self.prev_ll = None
        self._initialize_parameters()

    def parse(self):
        while self._should_continue():
            self._variational_e_step()
            self._variational_m_step()
        self._update_clusters()

    def provide_constraints(self, W):
        self.W = W

    def label_logs(self, log_labels):
        """
        log_labels: dictionary where each key is a true cluster and the values
                    are log indices.
        tokenized_logs: list of tokenized logs where the log_labels keys are a
                        subset.
        """
        for g, log_indices in enumerate(log_labels.values()):
            for log_idx in log_indices:
                x = self.C[log_idx]
                self.alpha[g] += 1
                self.beta[g, :] += x
                self.labeled_indices.append(log_idx)

    def _should_continue(self):
        if self.prev_ll is None:
            self.prev_ll = self._get_likelihood()
            return True
        ll = self._get_likelihood()
        improvement = abs((ll - self.prev_ll) / self.prev_ll)
        self.prev_ll = ll
        return self.epsilon < improvement

    def _get_likelihood(self):
        """
        Returns complete log-likelihood
        """
        log_likelihood = 0
        for n in range(self.N):
            x_n = self.C[n, :]
            g = self.R[n, :].argmax()
            pi_g = self.ex_ln_pi[g]
            theta_g = np.exp(self.ex_ln_theta[g])
            log_likelihood += (pi_g + log_multi(x_n, theta_g))
        return log_likelihood

    def _update_clusters(self):
        cluster_templates = {}
        for n in range(self.N):
            max_g = self.R[n, :].argmax()
            if max_g not in cluster_templates:
                cluster_templates[max_g] = []
            cluster_templates[max_g].append(n)
        self.cluster_templates = cluster_templates

    def _initialize_parameters(self):
        self._initialize_responsibilities()
        self._variational_m_step()

    def _variational_e_step(self):
        for g in range(self.G):
            ex_ln_pi_g = self.ex_ln_pi[g]
            ex_ln_theta_g = self.ex_ln_theta[g][np.newaxis, :]
            weight_term = self._get_weight_term(g)
            self.R[:, g] = (self.C * ex_ln_theta_g).sum(
                axis=1) + ex_ln_pi_g + weight_term
        self.R -= self.R.max(axis=1)[:, np.newaxis]
        self.R = np.exp(self.R)
        self.R /= self.R.sum(axis=1)[:, np.newaxis]

    def _variational_m_step(self):
        self.pi_v = self.R.sum(axis=0) + self.alpha
        self.ex_ln_pi = self._get_ex_ln(self.pi_v)
        for g in range(self.G):
            r_g = self.R[:, g][:, np.newaxis]
            self.theta_v[g, :] = (self.C * r_g).sum(axis=0) + self.beta[g]
            self.ex_ln_theta[g] = self._get_ex_ln(self.theta_v[g, :])

    def _get_ex_ln(self, params):
        return digamma(params) - digamma(params.sum())

    def _initialize_responsibilities(self):
        dir_params = self.alpha_0 * np.ones(self.G)
        for n in range(self.N):
            self.R[n, :] = np.random.dirichlet(dir_params)

    def _get_weight_term(self, g):
        weight_term = np.zeros(self.N)
        for n in self.W:
            for m in self.W[n]:
                weight_term[n] += self.W[n][m] * self.R[m, g]
        return weight_term
