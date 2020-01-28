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

    def evaluate(self, template_parsed):
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

    def get_type1_error_ratio(self):
        """
        Returns the ratio of log entries which belong to generated clusters
        that are composed of log entries from more than one true cluster.
        """
        error_line_count = 0
        for template_eval in self.template_evaluations:
            if len(template_eval.truth_templates) > 1:
                error_line_count += template_eval.actual_count
        return error_line_count / self.total_lines

    def get_type2_error_ratio(self):
        """
        Returns the ratio of log entries which belong to generated clusters
        that are all from the same true cluster and are missing at least one
        entry (i.e. incomplete clusters).
        """
        error_line_count = 0
        for template_eval in self.template_evaluations:
            if len(template_eval.truth_templates) == 1 and \
                    template_eval.actual_count != template_eval.expected_count:
                error_line_count += template_eval.actual_count
        return error_line_count / self.total_lines

    def _get_template_counts(self, parsed_entry_indices, truth_templates):
        truth_template_count = len(
            self.template_truth[truth_templates[0]])
        parsed_template_count = len(parsed_entry_indices)
        return parsed_template_count, truth_template_count

    def print_all_discrepancies(self, template_parsed):
        result = self.evaluate(template_parsed)
        for template_eval in self._get_specific_templates_evals(False):
            template_eval.print_discrepancies(template_parsed)
        print('Final accuracy: {}'.format(result))

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

    def _get_truth_templates(self, parsed_entry_indices):
        truth_templates = set()
        for idx in parsed_entry_indices:
            template = self.true_assignments[idx][-1]
            if template not in truth_templates:
                truth_templates.add(template)
        return list(truth_templates)

    def _get_template_truth(self, raw_truth):
        cluster_templates_truth = {}
        for raw_log_entry_truth in raw_truth[0:]:
            entry_id = raw_log_entry_truth[0]
            template = raw_log_entry_truth[-1]
            if template not in cluster_templates_truth:
                cluster_templates_truth[template] = []
            cluster_templates_truth[template].append(entry_id)
        return cluster_templates_truth
