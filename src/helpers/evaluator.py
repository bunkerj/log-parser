from src.utils import read_csv


class Evaluator:
    def __init__(self, true_assignments, parsed_results):
        self.true_assignments = true_assignments
        self.template_truth = self._get_template_truth(true_assignments)
        self.template_parsed = parsed_results
        self.total_lines = len(true_assignments)

    def evaluate(self):
        num_correct_lines = 0
        for template in self.template_parsed:
            parsed_entry_indices = self.template_parsed[template]
            line_count = len(parsed_entry_indices)
            truth_templates = self._get_truth_templates_from_parsed(parsed_entry_indices)
            if len(truth_templates) == 1:
                truth_templates_count = self._get_truth_templates_count(truth_templates)
                truth_template = list(truth_templates_count.keys())[0]
                if truth_templates_count[truth_template] == line_count:
                    num_correct_lines += line_count
        return num_correct_lines / self.total_lines

    def _get_truth_templates_from_parsed(self, parsed_entry_indices):
        truth_templates = set()
        for idx in parsed_entry_indices:
            template = self.true_assignments[idx][-1]
            if template not in truth_templates:
                truth_templates.add(template)
        return truth_templates

    def _get_truth_templates_count(self, truth_templates):
        truth_templates_count = {}
        for idx in range(len(self.true_assignments)):
            template = self.true_assignments[idx][-1]
            if template in truth_templates:
                if template not in truth_templates_count:
                    truth_templates_count[template] = 0
                truth_templates_count[template] += 1
        return truth_templates_count

    def _get_template_truth(self, raw_truth):
        cluster_templates_truth = {}
        for raw_log_entry_truth in raw_truth[1:]:
            entry_id = raw_log_entry_truth[0]
            template = raw_log_entry_truth[-1]
            if template not in cluster_templates_truth:
                cluster_templates_truth[template] = []
            cluster_templates_truth[template].append(entry_id)
        return cluster_templates_truth
