import numpy as np
from collections import defaultdict
from scipy.special import digamma, xlogy
from global_utils import log_multi_beta
from src.parsers.base.log_parser import LogParser
from src.utils import get_token_counts_batch, get_vocabulary_indices


class MultinomialMixtureVB(LogParser):
    def __init__(self, tokenized_logs, num_clusters,
                 epsilon=0.0001, max_iter=25):
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
        self.prev_elbo = None
        self.iter = 0
        self.max_iter = max_iter
        self._initialize_parameters()

    def parse(self):
        self.iter = 0
        self.prev_elbo = None
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
                self.beta[g] += x
                self.labeled_indices.append(log_idx)

    def _should_continue(self):
        """
        Check if another VB iteration should be performed.
        Update self.prev_ll and self.iter variables.
        """
        if self.iter > self.max_iter:
            return False
        elif self.prev_elbo is None:
            self.prev_elbo = self._get_elbo()
            return True
        elbo = self._get_elbo()
        improvement = abs((elbo - self.prev_elbo) / self.prev_elbo)
        if self.prev_elbo is not None:
            print(elbo - self.prev_elbo)
        self.prev_elbo = elbo
        self.iter += 1
        return self.epsilon < improvement

    def _get_elbo(self):
        return self._get_elbo_joint_term() - self._get_elbo_entropy_term()

    def _get_elbo_joint_term(self):
        joint_term = 0

        # TODO: see if there's a way to vectorize this more efficiently.
        for n in range(self.N):
            x_n = self.C[n].reshape(1, -1)
            for g in range(self.G):
                ex_ln_theta_g = self.ex_ln_theta[g].reshape(-1, 1)
                joint_term += self.R[n][g] * float(x_n @ ex_ln_theta_g)

        joint_term += (self.R @ self.ex_ln_pi.reshape(-1, 1)).sum()
        joint_term += ((self.alpha - 1) * self.ex_ln_pi).sum()
        joint_term += ((self.beta - 1) * self.ex_ln_theta).sum()

        return joint_term

    def _get_elbo_entropy_term(self):
        entropy_term = 0

        entropy_term += (xlogy(self.R, self.R)).sum()
        entropy_term += ((self.pi_v - 1) * self.ex_ln_pi).sum()
        entropy_term -= log_multi_beta(self.pi_v)
        entropy_term += ((self.theta_v - 1) * self.ex_ln_theta).sum()
        entropy_term -= log_multi_beta(self.theta_v).sum()

        return entropy_term

    def _update_clusters(self):
        cluster_templates = {}
        for n in range(self.N):
            max_g = self.R[n].argmax()
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
            ex_ln_theta_g = self.ex_ln_theta[g].reshape(1, -1)
            weight_term_g = self._get_weight_term(g)
            self.R[:, g] = (self.C * ex_ln_theta_g).sum(
                axis=1) + ex_ln_pi_g + weight_term_g
        self._norm_reponsibilities()

    def _variational_m_step(self):
        self.pi_v = self.R.sum(axis=0) + self.alpha
        self.ex_ln_pi = self._get_ex_ln(self.pi_v)
        for g in range(self.G):
            r_g = self.R[:, g].reshape(-1, 1)
            self.theta_v[g] = (self.C * r_g).sum(axis=0) + self.beta[g]
            self.ex_ln_theta[g] = self._get_ex_ln(self.theta_v[g])

    def _norm_reponsibilities(self):
        self.R -= self.R.max(axis=1).reshape(-1, 1)
        self.R = np.exp(self.R)
        self.R /= self.R.sum(axis=1).reshape(-1, 1)

    def _get_ex_ln(self, params):
        return digamma(params) - digamma(params.sum())

    def _initialize_responsibilities(self):
        dir_params = self.alpha_0 * np.ones(self.G)
        for n in range(self.N):
            self.R[n] = np.random.dirichlet(dir_params)

    def _get_weight_term(self, g):
        weight_term = np.zeros(self.N)
        for n in self.W:
            for m in self.W[n]:
                weight_term[n] += self.W[n][m] * self.R[m, g]
        return weight_term
