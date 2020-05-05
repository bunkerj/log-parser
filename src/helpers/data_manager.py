import re
from src.utils import read_csv
from global_constants import SPLIT_REGEX, PLACEHOLDER


def tokenize(template):
    return list(filter(lambda x: x != '', re.split(SPLIT_REGEX, template)))


class Template:
    def __init__(self, idx, template, regex):
        tokens = tokenize(template)
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
        self.headers, self.regex = self._generate_logformat_regex(data_config)

    def get_tokenized_logs(self, is_no_num=True):
        tokenized_logs = []
        for raw_log in self._get_raw_contents_from_unstructured():
            preprocessed_log = self._get_preprocessed_log(raw_log)
            tokenized_log = tokenize(preprocessed_log)
            if is_no_num:
                tokenized_log = self._get_tokenized_no_num_log(tokenized_log)
            tokenized_logs.append(tokenized_log)
        return tokenized_logs

    def get_tokenized_no_num_logs(self):
        # TODO: Replace this with get_tokenized_logs()
        logs = []
        for tokenized_logs in self.get_tokenized_logs():
            logs.append(self._get_tokenized_no_num_log(tokenized_logs))
        return logs

    def get_tokenized_log(self, raw_log_full_line, is_no_num=True):
        """
        Used for a streaming environment. Processes line from log file into a
        tokenized log. The is_no_num flag determines whether numbers are
        filtered out from the tokens.
        """
        preprocessed_log = self.get_preprocessed_log(raw_log_full_line)
        if preprocessed_log is not None:
            tokenized_log = tokenize(preprocessed_log)
            if is_no_num:
                return self._get_tokenized_no_num_log(tokenized_log)
            else:
                return tokenized_log
        return None

    def get_preprocessed_log(self, raw_log_full_line):
        raw_content = self._get_raw_content(raw_log_full_line)
        if raw_content is not None:
            return self._get_preprocessed_log(raw_content)
        return None

    def get_templates(self):
        raw_templates = read_csv(self.data_config['template_path'])
        templates = []
        for line in raw_templates[1:]:
            idx, template = line
            regex = self._get_template_regex(template)
            templates.append(Template(idx, template, regex))
        return templates

    def get_true_assignments(self):
        assignments_path = self.data_config['assignments_path']
        return read_csv(assignments_path)[1:]

    def _get_tokenized_no_num_log(self, tokenized_log):
        tokenized_no_num_log = []
        for token in tokenized_log:
            new_token = ''.join([c for c in token if not c.isdigit()])
            if new_token != '':
                tokenized_no_num_log.append(new_token)
        return tokenized_no_num_log

    def _get_template_regex(self, template):
        regex = re.escape(template)
        regex = re.sub(r'\\ \\\<\\\*\\\>$', r'\s?\S*', regex)
        regex = re.sub(r'\\\<\\\*\\\>', r'\s?\S*\s?', regex)
        regex = '^{}$'.format(regex)
        return regex

    def _get_preprocessed_log(self, raw_content):
        preprocessed_log = raw_content
        for currentRex in self.data_config['regex']:
            preprocessed_log = re.sub(currentRex, '', preprocessed_log)
        return preprocessed_log

    def _generate_logformat_regex(self, data_config):
        headers = []
        splitters = re.split(r'(<[^<>]+>)', data_config['log_format'])
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

    def _get_raw_contents_from_unstructured(self):
        log_contents = []
        log_file = self.data_config['unstructured_path']
        with open(log_file, 'r', encoding='utf-8') as f:
            for raw_log_full_line in f:
                raw_content = self._get_raw_content(raw_log_full_line)
                if raw_content is not None:
                    log_contents.append(raw_content)
        return log_contents

    def _get_raw_content(self, raw_log_full_line):
        match = self.regex.search(raw_log_full_line.strip())
        if match is None:
            return None
        return match.group('Content')
