from random import sample
from collections import defaultdict
from global_constants import MUST_LINK, CANNOT_LINK


class Oracle:
    def __init__(self, true_assignments):
        self.true_assignments = true_assignments
        self.true_clusters = defaultdict(list)
        self.true_references = []

    def get_constraints(self, parsed_clusters, n_samples_per_cluster,
                        tokenized_logs):
        split_parsed_clusters = self._get_split_clusters(parsed_clusters, True)
        split_true_clusters = self._get_split_clusters(parsed_clusters, False)

        cannot_link = self._get_links(split_parsed_clusters,
                                      n_samples_per_cluster,
                                      tokenized_logs)
        must_link = self._get_links(split_true_clusters,
                                    n_samples_per_cluster,
                                    tokenized_logs)

        return {CANNOT_LINK: cannot_link, MUST_LINK: must_link}

    def _get_links(self, split_clusters, n_samples_per_cluster, tokenized_logs):
        """
        Return constraints as a list of tuples where each
        tuple represents two logs that either should (true clusters split into
        parsed clusters) or should not (parsed clusters split into true
        clusters) be clustered together.
        """
        constraints = []
        err_clusters = [c for c in split_clusters.values() if len(c) > 1]
        for cluster in err_clusters:
            majority_event = self._get_majority_true_cluster(cluster)
            minority_events = [c for c in cluster if c != majority_event]

            majority_indices = cluster[majority_event]
            minority_event = sample(minority_events, 1)[0]
            minority_indices = cluster[minority_event]

            for _ in range(n_samples_per_cluster):
                sampled_maj_idx = sample(majority_indices, 1)[0]
                sampled_min_idx = sample(minority_indices, 1)[0]
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
