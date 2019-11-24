from src.utils import read_csv


class Evaluator:
    def __init__(self, data_config, parsed_results):
        raw_truth = read_csv(data_config['path'])
        self.raw_truth = raw_truth
        self.template_truth = self._get_template_truth(raw_truth)
        self.template_parsed = parsed_results
        self.total_lines = len(raw_truth) - 1

    def evaluate(self):
        num_correct_lines = 0
        for template in self.template_parsed:
            parsed_entry_indices = self.template_parsed[template]
            line_count = len(parsed_entry_indices)
            truth_templates = self._get_truth_templates(parsed_entry_indices)
            if len(truth_templates) == 1 and list(truth_templates.values())[0] == line_count:
                num_correct_lines += line_count
        return num_correct_lines / self.total_lines

    def _get_truth_templates(self, parsed_entry_indices):
        truth_templates = {}
        for idx in parsed_entry_indices:
            template = self.raw_truth[idx + 1][-1]
            if template not in truth_templates:
                truth_templates[template] = 0
            truth_templates[template] += 1
        return truth_templates

    def _get_template_truth(self, raw_truth):
        cluster_templates_truth = {}
        for raw_log_entry_truth in raw_truth[1:]:
            entry_id = raw_log_entry_truth[0]
            template = raw_log_entry_truth[-1]
            if template not in cluster_templates_truth:
                cluster_templates_truth[template] = []
            cluster_templates_truth[template].append(entry_id)
        return cluster_templates_truth
