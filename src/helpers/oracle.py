from random import sample, randint
from collections import defaultdict
from global_constants import MUST_LINK, CANNOT_LINK


class Oracle:
    def __init__(self, true_assignments):
        self.true_assignments = true_assignments
        self.true_clusters = defaultdict(list)
        self.true_references = []

    def get_corr_constraints(self, parsed_clusters, n_constraint_samples,
                             tokenized_logs, return_ids=False):
        """
        Return corrective constraints based on current parsed clustering.
        There are n_constraint_samples for cannot_link and must_link.
        """
        split_parsed_clusters = self._get_split_clusters(parsed_clusters, True)
        split_true_clusters = self._get_split_clusters(parsed_clusters, False)

        cannot_link = self._get_links(split_parsed_clusters,
                                      n_constraint_samples,
                                      tokenized_logs,
                                      return_ids)
        must_link = self._get_links(split_true_clusters,
                                    n_constraint_samples,
                                    tokenized_logs,
                                    return_ids)

        return {CANNOT_LINK: cannot_link, MUST_LINK: must_link}

    def get_constraints(self, n_constraint_samples):
        """
        Return n_constraints completely randomly sampled across all logs.
        There are n_constraint_samples total constraints.
        """
        cannot_link = []
        must_link = []
        random_pairs = self._get_random_pairs(n_constraint_samples)

        for pair in random_pairs:
            idx1, idx2 = pair
            g1 = self.true_assignments[idx1][-2]
            g2 = self.true_assignments[idx2][-2]
            if g1 != g2:
                cannot_link.append(pair)
            else:
                must_link.append(pair)

        return {CANNOT_LINK: cannot_link, MUST_LINK: must_link}

    def get_corr_constraints_matrix(self, parsed_clusters, n_constraint_samples,
                                    tokenized_logs, weight):
        constraints = self.get_corr_constraints(parsed_clusters,
                                                n_constraint_samples,
                                                tokenized_logs,
                                                True)
        return self._get_weight_matrix(constraints, weight)

    def get_constraints_matrix(self, n_constraint_samples, weight):
        constraints = self.get_constraints(n_constraint_samples)
        return self._get_weight_matrix(constraints, weight)

    def _get_weight_matrix(self, constraints, weight):
        W = defaultdict(dict)
        for idx1, idx2 in constraints[CANNOT_LINK]:
            W[idx1][idx2] = -weight
            W[idx2][idx1] = -weight
        for idx1, idx2 in constraints[MUST_LINK]:
            W[idx1][idx2] = weight
            W[idx2][idx1] = weight
        return W

    def _get_random_pairs(self, n_constraint_samples):
        N = len(self.true_assignments)
        return [(randint(0, N - 1), randint(0, N - 1)) for _ in
                range(n_constraint_samples)]

    def _get_links(self, split_clusters, n_constraint_samples, tokenized_logs,
                   return_ids):
        """
        Return constraints as a list of tuples where each
        tuple represents two logs that either should (true clusters split into
        parsed clusters) or should not (parsed clusters split into true
        clusters) be clustered together.
        """
        constraints = []
        err_clusters = [c for c in split_clusters.values() if len(c) > 1]

        if len(err_clusters) == 0:
            return []

        for _ in range(n_constraint_samples):
            cluster = sample(err_clusters, 1)[0]

            first_event = sample(cluster.keys(), 1)[0]
            second_events = [c for c in cluster if c != first_event]

            first_indices = cluster[first_event]
            second_event = sample(second_events, 1)[0]
            second_indices = cluster[second_event]

            sampled_maj_idx = sample(first_indices, 1)[0]
            sampled_min_idx = sample(second_indices, 1)[0]

            if return_ids:
                constraints.append((sampled_maj_idx, sampled_min_idx))
            else:
                constraints.append((tokenized_logs[sampled_maj_idx],
                                    tokenized_logs[sampled_min_idx]))

        return constraints

    def _get_split_clusters(self, parsed_clusters, is_parsed_split):
        """
        Returns a dictionary where each key corresponds to a cluster and
        the values are dictionaries each of which contains the indices for each
        subcluster found in the cluster. Note that labeled indices are
        completely ignored.

        If is_parsed_split is True, then each parsed cluster is given in terms
        of the true clusters. If False, then each true cluster is given in terms
        of the parsed clusters.
        """
        clusters = parsed_clusters if is_parsed_split \
            else self._get_true_clusters()
        references = self._get_true_references() if is_parsed_split \
            else self._get_parsed_references(parsed_clusters)
        split_clusters = {}
        for event in clusters:
            split_clusters[event] = defaultdict(list)
            for log_idx in clusters[event]:
                subcluster = references[log_idx]
                split_clusters[event][subcluster].append(log_idx)
        return split_clusters

    def _get_true_clusters(self):
        """
        Returns a dictionary where the keys are true cluster events and the
        values are list of log indices.
        """
        if len(self.true_clusters) != 0:
            return self.true_clusters

        for log_idx in range(len(self.true_assignments)):
            event = self.true_assignments[log_idx][-2]
            self.true_clusters[event].append(log_idx)

        return self.true_clusters

    def _get_true_references(self):
        """
        Returns a list where each index corresponds to a log index and the value
        corresponds to a true event.
        """
        if len(self.true_references) != 0:
            return self.true_references

        self.true_references = [a[-2] for a in self.true_assignments]
        return self.true_references

    def _get_parsed_references(self, parsed_clusters):
        """
        Returns a list where each index corresponds to a log index and the value
        corresponds to a parsed event.
        """
        parsed_reference = [0] * len(self.true_assignments)
        for event in parsed_clusters:
            for log_idx in parsed_clusters[event]:
                parsed_reference[log_idx] = event
        return parsed_reference

    def _get_majority_true_cluster(self, split_parsed_cluster):
        majority_event = None
        majority_count = 0
        for true_subcluster in split_parsed_cluster:
            n = len(split_parsed_cluster[true_subcluster])
            if n > majority_count:
                majority_event = true_subcluster
                majority_count = n
        return majority_event
