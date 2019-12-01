import re
import pandas as pd

SPLIT_REGEX = r'[\s=:,]'


class DataManager:
    def __init__(self, data_config):
        self.data_config = data_config

    def get_tokenized_log_entries(self):
        headers, regex = self._generate_logformat_regex()
        log_df = self._log_to_dataframe(self.data_config['unstructured_path'], regex, headers)
        return self._prepreprocess_raw_log_entries(log_df)

    def _prepreprocess_raw_log_entries(self, logdf):
        tokenized_log_entries = []
        for raw_log_msg in logdf['Content']:
            for currentRex in self.data_config['regex']:
                raw_log_msg = re.sub(currentRex, '', raw_log_msg)
            # log_entry = raw_log_msg.strip().split()
            log_entry = list(filter(lambda x: x != '', re.split(SPLIT_REGEX, raw_log_msg)))
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
        with open(log_file, 'r', encoding='utf-8') as fin:
            for line in fin.readlines():
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
        log_df = pd.DataFrame(log_messages, columns=headers)
        log_df.insert(0, 'LineId', None)
        log_df['LineId'] = [i + 1 for i in range(linecount)]
        return log_df
