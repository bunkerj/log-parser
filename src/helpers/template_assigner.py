import re
from global_constants import PLACEHOLDER
from src.helpers.data_manager import DataManager


class TemplateAssigner:
    def __init__(self, data_config):
        self.data_config = data_config
        self.data_manager = DataManager(data_config)

    def write_assignments(self):
        templates = self._get_templates()
        assignment_path = self.data_config['assignments_path']
        unstruct_path = self.data_config['unstructured_path']
        with open(unstruct_path, encoding='utf-8') as f_read:
            with open(assignment_path, 'w+', encoding='utf-8') as f_write:
                f_write.write('{}\n'.format('LineId,EventTemplate'))
                for idx, raw_log in enumerate(f_read.readlines(), start=1):
                    self._print_status(idx)
                    log = self.data_manager.process_raw_log(raw_log, False)
                    if log is not None:
                        match_idx = self._get_template_idx(log, templates)
                        f_write.write('{},{}\n'.format(idx, match_idx))

    def _print_status(self, idx):
        if idx % 100000 == 0:
            print('{}...'.format(idx))

    def _get_templates(self):
        templates = self.data_manager.get_templates()
        return sorted(templates,
                      reverse=True,
                      key=lambda t: self._get_const_count(t.tokens))

    def _get_template_idx(self, tokenized_log, templates):
        # TODO: Verify correctness and efficiency
        log = ' '.join(tokenized_log)
        for template in templates:
            if re.match(template.regex, log):
                return template.idx
        return -1

    def _get_const_count(self, tokenized_template):
        return sum(map(lambda token: token != PLACEHOLDER, tokenized_template))
