import re
import pandas as pd
from constants import SPLIT_REGEX, PLACEHOLDER
from src.utils import read_csv


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

    def get_tokenized_log_entries(self):
        log_df = self._get_log_dataframe()
        return self._preprocess_raw_log_entries(log_df)

    def get_tokenized_no_num_log_entries(self):
        tokenized_log_entries = self.get_tokenized_log_entries()
        tokenized_no_num_log_entries = []
        for tokenized_log_entry in tokenized_log_entries:
            tokenized_no_num_log_entry = []
            for token in tokenized_log_entry:
                new_token = ''.join([c for c in token if not c.isdigit()])
                if new_token != '':
                    tokenized_no_num_log_entry.append(new_token)
            tokenized_no_num_log_entries.append(tokenized_no_num_log_entry)
        return tokenized_no_num_log_entries

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

    def print_select_raw_and_tokenized_log_entries(self, log_indices):
        log_df = self._get_log_dataframe()
        raw_log_entries = log_df['Content'].to_list()
        tokenized_log_entries = self._preprocess_raw_log_entries(log_df)

        for idx in log_indices:
            print('Log Id: {}'.format(idx))
            print('\t{}'.format(raw_log_entries[idx]))
            print('\t{}\n'.format(tokenized_log_entries[idx]))

    def _get_log_dataframe(self):
        headers, regex = self._generate_logformat_regex()
        log_file = self.data_config['unstructured_path']
        return self._log_to_dataframe(log_file, regex, headers)

    def _get_template_regex(self, template):
        regex = re.escape(template)
        regex = re.sub(r'\\ \\\<\\\*\\\>$', r'\s?\S*', regex)
        regex = re.sub(r'\\\<\\\*\\\>', r'\s?\S*\s?', regex)
        regex = '^{}$'.format(regex)
        return regex

    def _preprocess_raw_log_entries(self, logdf):
        tokenized_log_entries = []
        for raw_log_msg in logdf['Content']:
            for currentRex in self.data_config['regex']:
                # # For Drain consistency
                # raw_log_msg = re.sub(currentRex, PLACEHOLDER, raw_log_msg)
                raw_log_msg = re.sub(currentRex, '',
                                     raw_log_msg)  # For IPLoM consistency
            # log_entry = raw_log_msg.strip().split()  # For Drain consistency
            log_entry = get_split_list(raw_log_msg)
            tokenized_log_entries.append(log_entry)
        return tokenized_log_entries

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

    def _log_to_dataframe(self, log_file, regex, headers):
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
