import re
from constants import PLACEHOLDER
from src.utils import read_csv
from abc import ABC, abstractmethod


class LogParser(ABC):
    def __init__(self, data_config):
        self.cluster_templates = {}
        self.regex_list = data_config['regex']
        self.log_file_path = data_config['path']
        self.tokenized_log_entries = self._get_tokenized_log_entries(data_config['path'])

    @abstractmethod
    def parse(self):
        pass

    def _get_tokenized_log_entries(self, structured_log_file_path):
        raw_log = read_csv(structured_log_file_path)
        result = []
        for structured_log_line in raw_log[1:]:
            raw_log_msg = structured_log_line[-3]
            for currentRex in self.regex_list:
                raw_log_msg = re.sub(currentRex, PLACEHOLDER, raw_log_msg)
            log_entry = raw_log_msg.strip().split()
            result.append(log_entry)
        return result
