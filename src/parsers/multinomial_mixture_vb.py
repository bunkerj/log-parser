import numpy as np
from collections import defaultdict
from scipy.special import digamma, xlogy
from global_utils import log_multi_beta, unnorm_log_multi, multi
from src.utils import get_token_counts_batch, get_vocabulary_indices


class MultinomialMixtureVB:
    def __init__(self):
        self.tokenized_logs = []
        self.epsilon = 0
        self.v_indices = {}
        self.C = np.array([]).reshape(-1, 1)
        self.G = 0
        self.N = len(self.tokenized_logs)
        self.V = len(self.v_indices)
        self.R = np.zeros((self.N, self.G))
        self.pi_v = np.zeros(self.G)
        self.theta_v = np.zeros((self.G, self.V))
        self.ex_ln_pi = np.zeros(self.G)
        self.ex_ln_theta = np.zeros((self.G, self.V))
        self.alpha_0 = 1 / self.G if self.G > 0 else 0
        self.beta_0 = 1 / self.V if self.V > 0 else 0
        self.alpha = self.alpha_0 + np.zeros(self.G)
        self.beta = self.beta_0 + np.zeros((self.G, self.V))
        self.labeled_indices = []
        self.W = defaultdict(dict)
        self.prev_elbo = None
        self.iter = 0
        self.max_iter = 999

    def fit(self, logs, num_clusters, log_labels=None,
            constraints=None, epsilon=0.0001, max_iter=25):
        """
        Fits variational parameters with respect to the provided logs.
        Returns the predictions for the log data used for fitting.
        """
        self._init_fields(logs, num_clusters, constraints, epsilon, max_iter)
        self._label_logs(log_labels)
        self._sample_parameters()
        self._run_variational_bayes()

    def predict(self, logs):
        """
        Returns the predictions for a specified set of log data that could be
        different than what was used to fit the model.
        """
        C = get_token_counts_batch(logs, self.v_indices)
        cluster_templates = defaultdict(list)
        for n in range(len(logs)):
            max_g = self._get_cluster_membership(C[n])
            cluster_templates[max_g].append(n)
        return cluster_templates

    def get_labeled_indices(self):
        return self.labeled_indices

    def _predict_with_current_responsibilities(self):
        cluster_templates = defaultdict(list)
        for n in range(self.N):
            max_g = self.R[n].argmax()
            cluster_templates[max_g].append(n)
        return cluster_templates

    def _get_cluster_membership(self, x_flat):
        r = np.array([self._get_unnorm_log_r(x_flat, g) for g in range(self.G)])
        return np.array(r).argmax()

    def _get_unnorm_log_r(self, x_flat, g):
        vocab_dist_term = unnorm_log_multi(x_flat, self.theta_v[g])
        cluster_dist_term = np.log(self.pi_v[g])
        return vocab_dist_term + cluster_dist_term

    def _run_variational_bayes(self):
        self.iter = 0
        self.prev_elbo = None
        while self._should_continue():
            self._variational_e_step()
            self._variational_m_step()

    def _label_logs(self, log_labels):
        """
        log_labels: dictionary where each key is a true cluster and the values
                    are log indices.
        tokenized_logs: list of tokenized logs where the log_labels keys are a
                        subset.
        """
        if log_labels is None:
            return
        for g, log_indices in enumerate(log_labels.values()):
            for log_idx in log_indices:
                x = self.C[log_idx]
                self.alpha[g] += 1
                self.beta[g] += x
                self.labeled_indices.append(log_idx)

    def _init_fields(self, tokenized_logs, num_clusters, constraints,
                     epsilon, max_iter):
        self.tokenized_logs = tokenized_logs
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
        self.alpha_0 = 1 / self.G
        self.beta_0 = 1 / self.V
        self.alpha = self.alpha_0 + np.zeros(self.G)
        self.beta = self.beta_0 + np.zeros((self.G, self.V))
        self.labeled_indices = []
        self.W = constraints or defaultdict(dict)
        self.prev_elbo = None
        self.iter = 0
        self.max_iter = max_iter

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
        improvement = (elbo - self.prev_elbo) / self.prev_elbo
        self.prev_elbo = elbo
        self.iter += 1
        return self.epsilon < improvement

    def _get_elbo(self):
        return self._get_elbo_joint_term() - self._get_elbo_entropy_term()

    def _get_elbo_joint_term(self):
        joint_term = 0
        joint_term += (self.C @ self.ex_ln_theta.T * self.R).sum()
        joint_term += (self.R @ self.ex_ln_pi.reshape(-1, 1)).sum()
        joint_term += ((self.alpha - 1) * self.ex_ln_pi).sum()
        joint_term += ((self.beta - 1) * self.ex_ln_theta).sum()
        joint_term += self._get_elbo_weight_penalty_term()
        return joint_term

    def _get_elbo_weight_penalty_term(self):
        penalty_term = 0
        for n in self.W:
            for m in self.W[n]:
                r = sum([self.R[n, g] * self.R[m, g] for g in range(self.G)])
                penalty_term += r * self.W[n][m]
        return penalty_term

    def _get_elbo_entropy_term(self):
        entropy_term = 0
        entropy_term += (xlogy(self.R, self.R)).sum()
        entropy_term += ((self.pi_v - 1) * self.ex_ln_pi).sum()
        entropy_term -= log_multi_beta(self.pi_v)
        entropy_term += ((self.theta_v - 1) * self.ex_ln_theta).sum()
        entropy_term -= log_multi_beta(self.theta_v).sum()
        return entropy_term

    def _sample_parameters(self):
        self._initialize_responsibilities()
        self._variational_m_step()

    def _variational_e_step(self):
        self.R = self._get_weight_penalty(self.R)
        self.R += self.C @ self.ex_ln_theta.T
        self.R += self.ex_ln_pi.reshape(1, -1)
        self._norm_responsibilities()

    def _variational_m_step(self):
        self.pi_v = self.R.sum(axis=0) + self.alpha
        self.ex_ln_pi = self._get_ex_ln(self.pi_v)
        self.theta_v = (self.R.T @ self.C) + self.beta
        self.ex_ln_theta = self._get_ex_ln(self.theta_v)

    def _get_weight_penalty(self, R):
        W_p = np.zeros(R.shape)
        for n in self.W:
            for g in range(self.G):
                W_p[n, g] += sum(self.W[n][m] * R[m, g] for m in self.W[n])
        return W_p

    def _norm_responsibilities(self):
        self.R -= self.R.max(axis=1).reshape(-1, 1)
        self.R = np.exp(self.R)
        self.R /= self.R.sum(axis=1).reshape(-1, 1)

    def _get_ex_ln(self, params):
        axis = params.ndim - 1
        params_sum = params.sum(axis=axis, keepdims=True)
        return digamma(params) - digamma(params_sum)

    def _initialize_responsibilities(self):
        dir_params = self.alpha_0 * np.ones(self.G)
        for n in range(self.N):
            self.R[n] = np.random.dirichlet(dir_params)
