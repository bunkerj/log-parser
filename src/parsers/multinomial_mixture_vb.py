import numpy as np
from scipy.special import digamma
from src.parsers.base.log_parser import LogParser
from src.utils import get_token_counts_batch, get_vocabulary_indices


class MultinomialMixtureVB(LogParser):
    def __init__(self, tokenized_logs, num_clusters, n_iter=20):
        super().__init__(tokenized_logs)
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
        self.n_iter = n_iter
        self.cluster_templates = {}
        self.alpha = 1 / self.G
        self.beta = 1 / self.V

    def parse(self):
        self._initialize_procedure()
        for _ in range(self.n_iter):
            self._variational_e_step()
            self._variational_m_step()
        self._update_clusters()

    def _update_clusters(self):
        cluster_templates = {}
        for n in range(self.N):
            max_g = self.R[n, :].argmax()
            if max_g not in cluster_templates:
                cluster_templates[max_g] = []
            cluster_templates[max_g].append(n)
        self.cluster_templates = cluster_templates

    def _initialize_procedure(self):
        self._initialize_responsibilities()
        self._variational_m_step()

    def _variational_e_step(self):
        for g in range(self.G):
            ex_ln_pi_g = self.ex_ln_pi[g]
            ex_ln_theta_g = self.ex_ln_theta[g, :][np.newaxis, :]
            self.R[:, g] = (self.C * ex_ln_theta_g).sum(axis=1) + ex_ln_pi_g
        self.R = np.exp(self.R)
        self.R /= self.R.sum(axis=1)[:, np.newaxis]

    def _variational_m_step(self):
        self.pi_v = self.R.sum(axis=0) + self.alpha
        self.ex_ln_pi = self._get_ex_ln(self.pi_v)
        for g in range(self.G):
            r_g = self.R[:, g][:, np.newaxis]
            self.theta_v[g, :] = (self.C * r_g).sum(axis=0) + self.beta
            self.ex_ln_theta[g] = self._get_ex_ln(self.theta_v[g, :])

    def _get_ex_ln(self, params):
        return digamma(params) - digamma(params.sum())

    def _initialize_responsibilities(self):
        dir_params = self.alpha * np.ones(self.G)
        for n in range(self.N):
            self.R[n, :] = np.random.dirichlet(dir_params)
