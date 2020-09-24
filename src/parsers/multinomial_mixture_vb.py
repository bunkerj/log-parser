import numpy as np
from collections import defaultdict
from scipy.special import digamma, xlogy
from global_utils import log_multi_beta
from src.utils import get_token_counts_batch, get_vocabulary_indices

LOG_LABELS_DEF = None
CS_WEIGHTS_DEF = None
P_WEIGHTS_DEF = None
EPSILON_DEF = 0.0001
MAX_ITER_DEF = 50


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
        self.p_weights = defaultdict(dict)
        self.prev_elbo = None
        self.iter = 0
        self.max_iter = 999

    def fit(self, logs, num_clusters, log_labels=LOG_LABELS_DEF,
            cs_weights=CS_WEIGHTS_DEF, p_weights=P_WEIGHTS_DEF,
            epsilon=EPSILON_DEF, max_iter=MAX_ITER_DEF):
        """
        Fits variational parameters with respect to the provided logs.
        Returns the predictions for the log data used for fitting.
        """
        self.init(logs, num_clusters, log_labels, cs_weights,
                  p_weights, epsilon, max_iter)
        self._run_variational_bayes()

    def fit_single_iter(self):
        self._variational_e_step()
        self._variational_m_step()

    def init(self, logs, num_clusters, log_labels=LOG_LABELS_DEF,
             cs_weights=CS_WEIGHTS_DEF, p_weights=P_WEIGHTS_DEF,
             epsilon=EPSILON_DEF, max_iter=MAX_ITER_DEF):
        self._init_fields(logs, num_clusters, cs_weights,
                          p_weights, epsilon, max_iter)
        self._label_logs(log_labels)
        self._sample_parameters()

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
        resps = np.zeros(self.G)
        for g in range(self.G):
            resps[g] = (x_flat * self.ex_ln_theta[g]).sum() + self.ex_ln_pi[g]
        return resps.argmax()

    def _run_variational_bayes(self):
        self.iter = 0
        self.prev_elbo = None
        while self._should_continue():
            self.fit_single_iter()

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

    def _init_fields(self, logs, num_clusters, cs_weights,
                     p_weights, epsilon, max_iter):
        self.tokenized_logs = logs
        self.epsilon = epsilon
        self.v_indices = get_vocabulary_indices(logs)
        self.C = get_token_counts_batch(logs, self.v_indices)
        self.G = num_clusters
        self.N = len(logs)
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
        self.cs_weights = self._init_cs_weights(cs_weights, self.N)
        self.p_weights = p_weights or defaultdict(dict)
        self.prev_elbo = None
        self.iter = 0
        self.max_iter = max_iter

    def _init_cs_weights(self, cs_weights, N):
        return np.ones(N) if cs_weights is None else np.array(cs_weights)

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
        for n in self.p_weights:
            for m in self.p_weights[n]:
                r = sum([self.R[n, g] * self.R[m, g] for g in range(self.G)])
                penalty_term += r * self.p_weights[n][m]
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
        self.R = self._get_penalty_weight_terms(self.R)
        self.R += self.C @ self.ex_ln_theta.T
        self.R += self.ex_ln_pi.reshape(1, -1)
        self.R *= self.cs_weights.reshape(-1, 1)
        self._norm_responsibilities()

    def _variational_m_step(self):
        self.pi_v = self.R.sum(axis=0) + self.alpha
        self.ex_ln_pi = self._get_ex_ln(self.pi_v)
        self.theta_v = (self.R.T @ self.C) + self.beta
        self.ex_ln_theta = self._get_ex_ln(self.theta_v)

    def _get_penalty_weight_terms(self, R):
        penalty_weight_terms = np.zeros(R.shape)
        for n in self.p_weights:
            for g in range(self.G):
                penalty_weight_terms[n, g] += self._get_p_weight_term(R, n, g)
        return penalty_weight_terms

    def _get_p_weight_term(self, R, n, g):
        return sum(self.p_weights[n][m] * R[m, g] for m in self.p_weights[n])

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
