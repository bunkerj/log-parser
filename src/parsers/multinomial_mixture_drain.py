import numpy as np
from random import sample
from src.parsers.drain import Drain

SAMPLE_SIZE = 5
N_QUERIES_PER_ROUND = 5
ZERO_THRESHOLD = 0.0000001


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
        self.frozen_indices = []
        self.R = None

    def parse(self):
        super().parse()
        self.R = self._get_initial_responsibilities()
        self._update_parameters()
        for i in range(N_QUERIES_PER_ROUND):
            self._run_em_procedure()
            self._merge_clusters()
            self._query_user()

    def _query_user(self):
        top_docs_to_query = self._get_top_resp_entropies(1)
        for d in top_docs_to_query:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            for g in range(self.num_clusters):
                print('---- Cluster {} ----'.format(g))
                if g in self.cluster_templates:
                    safe_sample_size = min(len(self.cluster_templates[g]),
                                           SAMPLE_SIZE)
                    samples_indices = sample(self.cluster_templates[g],
                                             safe_sample_size)
                    for idx in samples_indices:
                        print(' '.join(self.tokenized_log_entries[idx]))
                else:
                    print('empty')
                print()
            print('\n-------------------------------------------')
            print(' '.join(self.tokenized_log_entries[d]))
            print('-------------------------------------------\n')
            selected_cluster = int(input())
            self._apply_user_feedback(selected_cluster, d)

    def _get_top_resp_entropies(self, N):
        resp_entropies = np.apply_along_axis(self._get_entropy, 0, self.R)
        ordered_indices = np.argsort(-resp_entropies)
        return ordered_indices[:N]

    def _apply_user_feedback(self, selected_cluster, d):
        identical_token_indices = self._get_identical_token_indices(d)
        for d in identical_token_indices:
            self.R[:, d] = 0
            self.R[selected_cluster, d] = 1
            self.frozen_indices.append(d)

    def _get_identical_token_indices(self, d):
        identical_token_indices = []
        base_count_row = self.C[d, :]
        for d_comp in range(len(self.tokenized_log_entries)):
            count_row = self.C[d_comp, :]
            if np.all(base_count_row == count_row):
                identical_token_indices.append(d_comp)
        return identical_token_indices

    def _get_entropy(self, resp):
        filtered_resp = np.extract(resp > 0.1, resp)
        return sum(-r * np.log(r) for r in filtered_resp)

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
                # R[np.random.randint(0, self.num_clusters), log_idx] = 1
                R[cluster_idx, log_idx] = 1
            if cluster_idx < (G - 1):
                cluster_idx += 1
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
