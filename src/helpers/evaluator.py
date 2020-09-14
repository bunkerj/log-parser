import math
from sklearn.metrics import adjusted_mutual_info_score


class TemplateEvaluation:
    def __init__(self, parsed_template, actual_count):
        self.parsed_template = parsed_template
        self.truth_templates = []
        self.actual_count = actual_count
        self.expected_count = -1
        self.is_correct = True

    def update_truth_templates(self, truth_templates):
        self.truth_templates = truth_templates
        self.is_correct = len(truth_templates) == 1

    def update_expected_count(self, expected_count):
        self.expected_count = expected_count
        self.is_correct = self.actual_count == expected_count

    def print_discrepancies(self, template_parsed):
        print(self.parsed_template)
        print('Parsed log ids: {}'
              .format(template_parsed[self.parsed_template]))

        if len(self.truth_templates) != 1:
            print('Discrepancy: more than one true templates')
            for truth_template in self.truth_templates:
                print('\t{}'.format(truth_template))

        if self.actual_count != self.expected_count:
            print('Discrepancy: wrong true expected count')
            print('\tActual: {}   Expected: {}'.format(self.actual_count,
                                                       self.expected_count))
        print()


class Evaluator:
    def __init__(self, true_assignments):
        self.true_assignments = true_assignments
        self.template_truth = self._get_template_truth(true_assignments)
        self.total_lines = len(true_assignments)
        self.template_evaluations = []

    def get_accuracy(self, template_parsed):
        """
        Updates the template_evaluations field and return the ratio of correct
        lines.
        """
        self.template_evaluations = []
        for template, parsed_indices in template_parsed.items():
            actual_count = len(parsed_indices)
            template_eval = TemplateEvaluation(template, actual_count)
            truth_templates = self._get_truth_templates(parsed_indices)
            template_eval.update_truth_templates(truth_templates)
            if len(truth_templates) == 1:
                expected_count = len(self.template_truth[truth_templates[0]])
                template_eval.update_expected_count(expected_count)
            self.template_evaluations.append(template_eval)
        return self._get_ratio_of_correct_lines()

    def get_ami(self, parsed_clusters):
        true_references = [a[-2] for a in self.true_assignments]
        parsed_references = self._get_parsed_references(parsed_clusters)
        return adjusted_mutual_info_score(true_references, parsed_references)

    def get_type1_error_ratio(self):
        """
        Returns the ratio of logs which belong to generated clusters
        that are composed of logs from more than one true cluster.

        Must call get_accuracy() prior to using this function.
        """
        error_line_count = 0
        for template_eval in self.template_evaluations:
            if len(template_eval.truth_templates) > 1:
                error_line_count += template_eval.actual_count
        return error_line_count / self.total_lines

    def get_type2_error_ratio(self):
        """
        Returns the ratio of logs which belong to generated clusters
        that are all from the same true cluster and are missing at least one
        log (i.e. incomplete clusters).

        Must call get_accuracy() prior to using this function.
        """
        error_line_count = 0
        for template_eval in self.template_evaluations:
            if len(template_eval.truth_templates) == 1 and \
                    template_eval.actual_count != template_eval.expected_count:
                error_line_count += template_eval.actual_count
        return error_line_count / self.total_lines

    def get_impurity(self, template_parsed, labeled_indices):
        """
        Returns a measure of impurity using entropy.
        """
        N = len(self.true_assignments) - len(labeled_indices)
        if N == 0:
            return 0
        total_impurity = 0
        event_counts = self._get_event_counts(template_parsed, labeled_indices)
        for cluster in event_counts:
            entropy = 0
            cluster_size = sum(event_counts[cluster].values())
            true_cluster_count = len(event_counts[cluster])
            for event in event_counts[cluster]:
                count = event_counts[cluster][event]
                p = count / cluster_size
                entropy += -p * math.log(p)
            norm_entropy = self._get_norm_impurity(entropy, true_cluster_count)
            total_impurity += (cluster_size / N) * norm_entropy
        return total_impurity

    def print_all_discrepancies(self, template_parsed):
        result = self.get_accuracy(template_parsed)
        for template_eval in self._get_specific_templates_evals(False):
            template_eval.print_discrepancies(template_parsed)
        print('Final accuracy: {}'.format(result))

    def _get_parsed_references(self, parsed_clusters):
        """
        Returns a list where each index corresponds to a log index and the value
        corresponds to the index of a parsed event.
        """
        parsed_reference = [0] * len(self.true_assignments)
        for event_idx, event in enumerate(parsed_clusters):
            for log_idx in parsed_clusters[event]:
                parsed_reference[log_idx] = event_idx
        return parsed_reference

    def _get_norm_impurity(self, entropy, true_cluster_count):
        return entropy / math.log(true_cluster_count) \
            if true_cluster_count > 1 else entropy

    def _get_event_counts(self, template_parsed, labeled_indices):
        """
        Returns a dictionary where each key corresponds to a parsed cluster and
        the values are dictionaries each of which contains the counts for each
        true event found in the cluster. Note that labeled indices are
        completely ignored.
        """
        event_counts = {}
        for cluster in template_parsed:
            if cluster not in event_counts:
                event_counts[cluster] = {}
                for log_idx in template_parsed[cluster]:
                    if log_idx in labeled_indices:
                        continue
                    event = self.true_assignments[log_idx][-2]
                    if event not in event_counts[cluster]:
                        event_counts[cluster][event] = 0
                    event_counts[cluster][event] += 1
        return event_counts

    def _get_template_counts(self, parsed_log_indices, truth_templates):
        truth_template_count = len(
            self.template_truth[truth_templates[0]])
        parsed_template_count = len(parsed_log_indices)
        return parsed_template_count, truth_template_count

    def _get_ratio_of_correct_lines(self):
        correct_template_evals = self._get_specific_templates_evals(True)
        num_correct_lines = self._get_num_correct_lines(correct_template_evals)
        return num_correct_lines / self.total_lines

    def _get_specific_templates_evals(self, is_correct):
        return filter(lambda t_eval: is_correct == t_eval.is_correct,
                      self.template_evaluations)

    def _get_num_correct_lines(self, correct_templates):
        return sum([temp_eval.actual_count
                    for temp_eval in correct_templates])

    def _get_truth_templates(self, parsed_log_indices):
        truth_templates = set()
        for idx in parsed_log_indices:
            template = self.true_assignments[idx][-1]
            if template not in truth_templates:
                truth_templates.add(template)
        return list(truth_templates)

    def _get_template_truth(self, raw_truth):
        cluster_templates_truth = {}
        for log_id, raw_log_truth in enumerate(raw_truth):
            template = raw_log_truth[-1]
            if template not in cluster_templates_truth:
                cluster_templates_truth[template] = []
            cluster_templates_truth[template].append(log_id)
        return cluster_templates_truth
