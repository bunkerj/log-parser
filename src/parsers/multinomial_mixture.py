from src.parsers.log_parser import LogParser
import numpy as np

ZERO_THRESHOLD = 0.0000001


class MultinomialMixture(LogParser):
    def __init__(self, tokenized_log_entries, num_clusters):
        super().__init__(tokenized_log_entries)
        self.num_clusters = num_clusters
        self.v_indices = self._get_vocabulary_indices(tokenized_log_entries)
        self.C = self._get_counts(tokenized_log_entries)
        self.Pi = np.zeros((num_clusters, 1))
        self.Theta = np.zeros((num_clusters, len(self.v_indices)))
        self.frozen_indices = []
        self.R = None

    def parse(self):
        self.R = self._get_initial_responsibilities()
        self._update_parameters()
        self._run_em_procedure()
        self._merge_clusters()

    def print_cluster_samples(self, n_samples):
        for cluster_idx in self.cluster_templates:
            print('Cluster {}'.format(cluster_idx))
            log_indices = self.cluster_templates[cluster_idx][:n_samples]
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
        self.Pi /= sum(self.Pi)
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
            print(likelihood)
            if old_likelihood is not None and \
                    (abs(old_likelihood - likelihood)
                     / old_likelihood) < ZERO_THRESHOLD:
                break
            old_likelihood = likelihood

    def _update_responsibilities(self):
        # TODO: Make this more efficient with masking
        G, D = self.R.shape
        new_R = np.zeros(self.R.shape)
        for g in range(G):
            for d in range(D):
                new_R[g, d] = self.Pi[g] * self._get_multinomial_term(g, d)
        for d in self.frozen_indices:
            new_R[:, d] = self.R[:, d]
        self.R = new_R
        self.R /= self.R.sum(axis=0, keepdims=True)

    def _get_multinomial_term(self, k, d):
        factorize = lambda arr: np.array([np.math.factorial(x) for x in arr])
        coeff_num = np.math.factorial(self.C[d, :].sum())
        coeff_den = factorize(self.C[d, :]).prod()
        coeff = coeff_num / coeff_den
        result = coeff * (self.Theta[k, :] ** self.C[d, :]).prod()
        return result

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
                likelihood += -99999999999
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
