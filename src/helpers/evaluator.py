from src.utils import read_csv


class Evaluator:
    def __init__(self, structured_path, parsed_results):
        raw_truth = read_csv(structured_path)
        self.raw_truth = raw_truth
        self.template_truth = self._get_template_truth(raw_truth)
        self.template_parsed = parsed_results

    def evaluate(self):
        # TODO: Implement percentage accuracy measure
        percentage_accuracy = 0

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
