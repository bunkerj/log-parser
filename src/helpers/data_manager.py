import re
import pandas as pd
from src.utils import read_csv
from global_constants import SPLIT_REGEX, PLACEHOLDER


def get_split_list(template):
    return list(filter(lambda x: x != '', re.split(SPLIT_REGEX, template)))


class Template:
    def __init__(self, idx, template, regex):
        tokens = get_split_list(template)
        self.idx = idx
        self.tokens = tokens
        self.regex = regex

    def _get_constant_tokens_indices(self, tokens):
        constant_tokens_indices = []
        for idx, token in enumerate(tokens):
            if token != PLACEHOLDER:
                constant_tokens_indices.append(idx)
        return constant_tokens_indices


class DataManager:
    def __init__(self, data_config):
        self.data_config = data_config

    def get_tokenized_logs(self):
        raw_logs = self._get_raw_logs()
        return self._preprocess_raw_logs(raw_logs)

    def get_tokenized_no_num_logs(self):
        logs = []
        for tokenized_logs in self.get_tokenized_logs():
            logs.append(self._get_tokenized_no_num_log(tokenized_logs))
        return logs

    def get_templates(self):
        raw_templates = read_csv(self.data_config['template_path'])
        templates = []
        for line in raw_templates[1:]:
            idx, template = line
            template = ' '.join(get_split_list(template))
            regex = self._get_template_regex(template)
            templates.append(Template(idx, template, regex))
        return templates

    def get_raw_log_full_lines(self):
        """
        Returns the raw full lines including but not limited to the message
        component.
        """
        headers, regex = self._generate_logformat_regex()
        log_file = self.data_config['unstructured_path']
        return self._get_raw_log_full_lines(log_file, regex, headers)

    def get_true_assignments(self):
        assignments_path = self.data_config['assignments_path']
        return read_csv(assignments_path)[1:]

    def print_select_raw_and_tokenized_logs(self, log_indices):
        raw_logs = self._get_raw_logs()
        tokenized_logs = self._preprocess_raw_logs(raw_logs)

        for idx in log_indices:
            print('Log Id: {}'.format(idx))
            print('\t{}'.format(raw_logs[idx]))
            print('\t{}\n'.format(tokenized_logs[idx]))

    def _get_tokenized_no_num_log(self, tokenized_log):
        tokenized_no_num_log = []
        for token in tokenized_log:
            new_token = ''.join([c for c in token if not c.isdigit()])
            if new_token != '':
                tokenized_no_num_log.append(new_token)
        return tokenized_no_num_log

    def _get_raw_logs(self):
        headers, regex = self._generate_logformat_regex()
        log_file = self.data_config['unstructured_path']
        return self._log_to_df(log_file, regex, headers)['Content'].to_list()

    def _get_template_regex(self, template):
        regex = re.escape(template)
        regex = re.sub(r'\\ \\\<\\\*\\\>$', r'\s?\S*', regex)
        regex = re.sub(r'\\\<\\\*\\\>', r'\s?\S*\s?', regex)
        regex = '^{}$'.format(regex)
        return regex

    def _preprocess_raw_logs(self, raw_logs):
        tokenized_logs = []
        for raw_log in raw_logs:
            log = self._preprocess_raw_log(raw_log)
            tokenized_logs.append(log)
        return tokenized_logs

    def _preprocess_raw_log(self, raw_log):
        for currentRex in self.data_config['regex']:
            # # For Drain consistency
            # raw_log_msg = re.sub(currentRex, PLACEHOLDER, raw_log_msg)
            raw_log = re.sub(currentRex, '',
                             raw_log)  # For IPLoM consistency
        # log = raw_log_msg.strip().split()  # For Drain consistency
        return get_split_list(raw_log)

    def _generate_logformat_regex(self):
        headers = []
        splitters = re.split(r'(<[^<>]+>)', self.data_config['log_format'])
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^{}$'.format(regex))
        return headers, regex

    def _log_to_df(self, log_file, regex, headers):
        log_messages = self._get_raw_log_full_lines(log_file, regex, headers)
        log_df = pd.DataFrame(log_messages, columns=headers)
        log_df.insert(0, 'LineId', None)
        log_df['LineId'] = [i + 1 for i in range(len(log_messages))]
        return log_df

    def _get_raw_log_full_lines(self, log_file, regex, headers):
        log_messages = []
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                match = regex.search(line.strip())
                if match is None:
                    continue
                message = [match.group(header) for header in headers]
                log_messages.append(message)
        return log_messages
