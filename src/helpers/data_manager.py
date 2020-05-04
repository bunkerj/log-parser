import re
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
        self.headers, self.regex = self._generate_logformat_regex(data_config)

    def get_tokenized_logs(self):
        raw_logs = self._get_raw_logs()
        return self._preprocess_raw_logs(raw_logs)

    def process_raw_log(self, raw_log_full_line, is_no_num=True):
        """
        Used for a streaming environment. Processes line from log file into a
        tokenized log. The is_no_num flag determines whether numbers are
        filtered out from the tokens.
        """
        raw_log_content = self.get_raw_log_content(raw_log_full_line)
        if raw_log_content is not None:
            tokenized_log = self._preprocess_raw_log(raw_log_content)
            if is_no_num:
                return self._get_tokenized_no_num_log(tokenized_log)
            else:
                return tokenized_log
        return None

    def get_raw_log_content(self, raw_log_full_line):
        log_full_line = self._get_log_full_line(raw_log_full_line)
        if log_full_line is not None:
            raw_log_contents = self._extract_contents([log_full_line])
            return raw_log_contents[0]
        return None

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
            # template = ' '.join(get_split_list(template))
            regex = self._get_template_regex(template)
            templates.append(Template(idx, template, regex))
        return templates

    def get_raw_log_full_lines(self):
        """
        Returns the raw full lines including but not limited to the message
        component.
        """
        log_file = self.data_config['unstructured_path']
        return self._get_raw_log_full_lines(log_file)

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
        log_file = self.data_config['unstructured_path']
        raw_log_full_lines = self._get_raw_log_full_lines(log_file)
        return self._extract_contents(raw_log_full_lines)

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

    def _preprocess_raw_log(self, raw_log_content):
        for currentRex in self.data_config['regex']:
            raw_log_content = re.sub(currentRex, '', raw_log_content)
        return get_split_list(raw_log_content)

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

    def _extract_contents(self, raw_logs):
        content_idx = self.headers.index('Content')
        return [raw_log[content_idx] for raw_log in raw_logs]

    def _get_raw_log_full_lines(self, log_file):
        processed_log_lines = []
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                processed_log_line = self._get_log_full_line(line)
                if processed_log_line is None:
                    continue
                processed_log_lines.append(processed_log_line)
        return processed_log_lines

    def _get_log_full_line(self, log_line):
        match = self.regex.search(log_line.strip())
        if match is None:
            return None
        # TODO: can probably use a dict here
        return [match.group(header) for header in self.headers]
