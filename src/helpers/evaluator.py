from src.utils import read_csv, are_lists_equal


class Evaluator:
    def __init__(self, structured_path, parsed_results):
        raw_truth = read_csv(structured_path)
        self.raw_truth = raw_truth
        self.template_truth = self._get_template_truth(raw_truth)
        self.template_parsed = parsed_results

    def evaluate(self):
        num_correct_lines = 0
        total_lines = 0
        matching_templates = self._get_matching_templates()

        if len(matching_templates) == 0:
            print('Warning: no templates were extracted')
            return -1

        for template in matching_templates:
            parsed_entry_indices = self.template_parsed[template]
            truth_entry_indices = self.template_truth[template]
            line_count = len(parsed_entry_indices)
            if are_lists_equal(parsed_entry_indices, truth_entry_indices):
                num_correct_lines += line_count
            total_lines += line_count

        return num_correct_lines / total_lines

    def _get_matching_templates(self):
        matching_templates = set()
        for template in self.template_parsed:
            if template in self.template_truth:
                matching_templates.add(template)
        return matching_templates

    def _get_template_truth(self, raw_truth):
        cluster_templates_truth = {}
        for raw_log_entry_truth in raw_truth[1:]:
            entry_id = raw_log_entry_truth[0]
            template = raw_log_entry_truth[-1]
            if template not in cluster_templates_truth:
                cluster_templates_truth[template] = []
            cluster_templates_truth[template].append(entry_id)
        return cluster_templates_truth

    def _get_percentage_accuracy(self):
        pass
