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
        self.constant_token_indices = self._get_constant_tokens_indices(tokens)

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
        headers, regex = self._generate_logformat_regex()
        log_df = self._log_to_dataframe(self.data_config['unstructured_path'], regex, headers)
        return self._prepreprocess_raw_log_entries(log_df)

    def get_templates(self):
        raw_templates = read_csv(self.data_config['template_path'])
        templates = []
        for line in raw_templates[1:]:
            idx, template = line
            template = ' '.join(get_split_list(template))
            regex = re.escape(template)
            regex = re.sub(r'^\\\<\\\*\\\>', r'\S*\s?', regex)
            regex = re.sub(r'\\ \\\<\\\*\\\>$', r'\s?\S*', regex)
            regex = re.sub(r'\\\<\\\*\\\>', r'\s?\S*\s?', regex)
            templates.append(Template(idx, template, regex))
        return templates

    def _prepreprocess_raw_log_entries(self, logdf):
        tokenized_log_entries = []
        for raw_log_msg in logdf['Content']:
            for currentRex in self.data_config['regex']:
                raw_log_msg = re.sub(currentRex, '', raw_log_msg)
            # log_entry = raw_log_msg.strip().split()
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
        regex = re.compile('^' + regex + '$')
        return headers, regex

    def _log_to_dataframe(self, log_file, regex, headers):
        log_messages = []
        linecount = 0
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                match = regex.search(line.strip())
                message = [match.group(header) for header in headers]
                log_messages.append(message)
                linecount += 1
                if not (0 < linecount <= 200000):
                    break
        log_df = pd.DataFrame(log_messages, columns=headers)
        log_df.insert(0, 'LineId', None)
        log_df['LineId'] = [i + 1 for i in range(linecount)]
        return log_df
