import re
from constants import PLACEHOLDER
from src.helpers.data_manager import DataManager


class TemplateAssigner:
    def __init__(self, data_config):
        self.data_config = data_config

    def write_assignments(self):
        template_assignments = self._get_template_assignments()
        with open(self.data_config['assignments_path'], 'w+', encoding='utf-8') as f:
            for template_assignment in template_assignments:
                f.write('{}\n'.format(template_assignment))

    def _get_template_assignments(self):
        data_manager = DataManager(self.data_config)
        tokenized_log_entries = data_manager.get_tokenized_log_entries()
        templates = data_manager.get_templates()
        sorted_templates = sorted(templates,
                                  reverse=True,
                                  key=lambda t: self._get_constant_count(t.tokens))
        assignments = []
        for log_idx, tokenized_log_entry in enumerate(tokenized_log_entries):
            if (log_idx + 1) % 100000 == 0:
                print('Log {}/{}...'.format(log_idx + 1, len(tokenized_log_entries)))
            match_idx = self._get_matching_template_idx(tokenized_log_entry, sorted_templates)
            assignments.append(match_idx)
        return assignments

    def _get_matching_template_idx(self, tokenized_log_entry, templates):
        # TODO: Verify correctness and efficiency
        for template in templates:
            log_entry = ' '.join(tokenized_log_entry)
            if re.match(template.regex, log_entry):
                return template.idx
        return -1

    def _get_constant_count(self, tokenized_template):
        return sum(map(lambda token: token != PLACEHOLDER, tokenized_template))
