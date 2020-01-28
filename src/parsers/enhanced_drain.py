from src.parsers.drain import Drain
from copy import deepcopy
import Levenshtein as lev


class EnhancedDrain(Drain):
    def __init__(self, tokenized_log_entries, max_depth, max_child,
                 sim_threshold, edit_ratio_threshold):
        super().__init__(tokenized_log_entries, max_depth, max_child,
                         sim_threshold)
        self.edit_ratio_threshold = edit_ratio_threshold

    def parse(self):
        super().parse()
        self._post_process_clusters()

    def _post_process_clusters(self):
        new_cluster_templates = deepcopy(self.cluster_templates)
        while True:
            merge_performed = False
            for t1 in new_cluster_templates:
                for t2 in new_cluster_templates:
                    if t1 != t2 and self._is_edit_ratio_significant(t1, t2):
                        self._merge_clusters(t1, t2, new_cluster_templates)
                        merge_performed = True
                    if merge_performed:
                        break
                if merge_performed:
                    break
            if not merge_performed:
                break
        self.cluster_templates = new_cluster_templates

    def _is_edit_ratio_significant(self, t1, t2):
        return lev.ratio(t1, t2) > self.edit_ratio_threshold

    def _merge_clusters(self, t1, t2, cluster_templates):
        median_t = lev.median([t1, t2])
        common_indices = sorted(cluster_templates[t1] + cluster_templates[t2])
        del cluster_templates[t1]
        del cluster_templates[t2]
        cluster_templates[median_t] = common_indices
