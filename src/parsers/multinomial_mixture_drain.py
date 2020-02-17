import numpy as np
from src.parsers.drain import Drain


class MultinomialMixtureDrain(Drain):
    def __init__(self, tokenized_log_entries, max_depth, max_child,
                 sim_threshold, num_clusters):
        super().__init__(tokenized_log_entries, max_depth, max_child,
                         sim_threshold)
        self.num_clusters = num_clusters
        self.v_indices = self._get_vocabulary_indices(tokenized_log_entries)
        self.C = self._get_counts(tokenized_log_entries)
        self.Pi = np.zeros((num_clusters, 1))
        self.Theta = np.zeros((num_clusters, len(self.v_indices)))
        self.R = None

    def parse(self):
        super().parse()
        self.R = self._get_initial_responsibilities()
        self._update_parameters()
        self._run_em_procedure()
        self._merge_clusters()

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
        cluster_idx = 0
        for cluster in self.cluster_templates:
            log_entry_indices = self.cluster_templates[cluster]
            for log_idx in log_entry_indices:
                R[cluster_idx, log_idx] = 1
            if cluster_idx < (G - 1):
                cluster_idx += 1
        return R

    def _update_parameters(self):
        self.Pi = self.R.sum(axis=1, keepdims=True)
        self.Pi /= sum(self.Pi)
        self.Theta = self.R @ self.C
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
                    abs(old_likelihood - likelihood) < 0.0000001:
                break
            old_likelihood = likelihood

    def _update_responsibilities(self):
        G, D = self.R.shape
        for g in range(G):
            for d in range(D):
                self.R[g, d] = self.Pi[g] * self._get_multinomial_term(g, d)
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
            likelihood += np.log(sum_term)
        return float(likelihood)

    def _merge_clusters(self):
        new_cluster_templates = {}
        D = self.R.shape[1]
        for d in range(D):
            cluster_idx = self.R[:, d].argmax()
            if cluster_idx not in new_cluster_templates:
                new_cluster_templates[cluster_idx] = []
            new_cluster_templates[cluster_idx].append(d)
        for g in list(new_cluster_templates):
            log_idx = new_cluster_templates[g][0]
            pseudo_template = ' '.join(self.tokenized_log_entries[log_idx])
            new_cluster_templates[pseudo_template] = new_cluster_templates[g]
            del new_cluster_templates[g]
        self.cluster_templates = new_cluster_templates
