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
        unmatched_templates = set()

        templates_tmp = [t.tokens for t in templates]

        for log_idx, tokenized_log_entry in enumerate(tokenized_log_entries):
            match_idx = self._get_matching_template_idx(tokenized_log_entry, sorted_templates)
            assignments.append(match_idx)
            if match_idx == -1:
                log_entry_str = ' '.join(tokenized_log_entry)
                print('Could not match: {}'.format(log_entry_str))
                unmatched_templates.add(' '.join(tokenized_log_entry))
        return assignments

    def _get_matching_template_idx(self, tokenized_log_entry, templates):
        # TODO: Verify correctness and efficiency
        for template in templates:
            log_entry = ' '.join(tokenized_log_entry)
            if re.match(template.regex, log_entry):
                return template.idx
            else:
                return -1
        # for idx in range(len(tokenized_log_entry)):
        #     possible_templates = list(filter(lambda t: self._is_strong_match(t, tokenized_log_entry), templates))
        #     if len(possible_templates) > 0:
        #         matched_template = self._get_template_with_lowest_placeholder_count(possible_templates)
        #         return matched_template.idx
        #     else:
        #         possible_templates = list(filter(lambda t: self._is_weak_match(t, tokenized_log_entry), templates))
        #         if len(possible_templates) > 0:
        #             matched_template = self._get_template_with_lowest_placeholder_count(possible_templates)
        #             log_entry_str = ' '.join(tokenized_log_entry)
        #             matched_template_str = ' '.join(matched_template.tokens)
        #             print('Weak match: {} -----> {}'.format(log_entry_str, matched_template_str))
        #             return matched_template.idx
        #         else:
        #             raise Exception('Cannot find a matching template')

    def _is_weak_match(self, template, tokenized_log_entry):
        for idx in template.constant_token_indices:
            if template.tokens[idx] not in tokenized_log_entry:
                return False
        return True

    def _is_strong_match(self, template, tokenized_log_entry):
        if len(tokenized_log_entry) > len(template.tokens):
            return False
        for idx in template.constant_token_indices:
            if idx < len(tokenized_log_entry) and tokenized_log_entry[idx] != template.tokens[idx]:
                return False
        return True

    def _get_template_with_lowest_placeholder_count(self, remaining_templates):
        lowest_placeholder_count = len(remaining_templates[0].tokens)
        best_template = remaining_templates[0]
        for template in remaining_templates:
            placeholder_count = self._get_placeholder_count(template.tokens)
            if placeholder_count < lowest_placeholder_count:
                lowest_placeholder_count = placeholder_count
                best_template = template
        return best_template

    def _get_placeholder_count(self, tokenized_template):
        return sum(map(lambda token: token == PLACEHOLDER, tokenized_template))

    def _get_constant_count(self, tokenized_template):
        return sum(map(lambda token: token != PLACEHOLDER, tokenized_template))
