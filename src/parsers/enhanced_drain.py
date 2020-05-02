import numpy as np
import Levenshtein as lev
from src.parsers.drain import Drain
from src.utils import plot_dendrogram
from sklearn.cluster import AgglomerativeClustering


class EnhancedDrain(Drain):
    def __init__(self, tokenized_logs, max_depth, max_child,
                 sim_threshold, edit_ratio_threshold):
        super().__init__(tokenized_logs, max_depth, max_child,
                         sim_threshold)
        self.edit_ratio_threshold = edit_ratio_threshold
        self.agglo_clustering_model = None
        self.original_labels = None

    def parse(self):
        super().parse()
        self._post_process_clusters()

    def plot_dendrogram(self, **kwargs):
        plot_dendrogram(self.agglo_clustering_model,
                        labels=self.original_labels,
                        **kwargs)

    def _post_process_clusters(self):
        cluster_template_strings = self._get_cluster_template_strings()
        distance_matrix = self._get_distance_matrix(cluster_template_strings)
        self.original_labels = self._get_cluster_template_strings()
        self.agglo_clustering_model = AgglomerativeClustering(
            distance_threshold=self.edit_ratio_threshold,
            affinity='precomputed',
            linkage='single',
            n_clusters=None,
        ).fit(distance_matrix)
        self._merge_clusters()

    def _get_cluster_template_strings(self):
        return list(self.cluster_templates.keys())

    def _get_distance_matrix(self, values):
        n = len(values)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                distance_matrix[i][j] = 1 - lev.ratio(values[i], values[j])
                distance_matrix[j][i] = 1 - lev.ratio(values[i], values[j])
        return distance_matrix

    def _merge_clusters(self):
        labels = list(self.agglo_clustering_model.labels_)
        cluster_templates = {}
        log_indices = {}

        for idx, template in enumerate(self.cluster_templates):
            label = labels[idx]
            if label not in cluster_templates:
                cluster_templates[label] = []
                log_indices[label] = []
            cluster_templates[label].append(template)
            log_indices[label] += self.cluster_templates[template]

        self.cluster_templates = \
            self._get_new_cluster_templates(labels,
                                            cluster_templates,
                                            log_indices)

    def _get_new_cluster_templates(self, labels, cluster_templates,
                                   log_indices):
        new_cluster_templates = {}
        for label in labels:
            new_cluster_template = lev.median(cluster_templates[label])
            new_cluster_templates[new_cluster_template] = sorted(
                log_indices[label])
        return new_cluster_templates
