from src.utils import read_csv


class LogParser:
    def __init__(self, log_file_path):
        self.cluster_templates = {}
        self.log_file_path = log_file_path
        self.tokenized_log_entries = self._get_tokenized_log_entries(log_file_path)

    def _get_tokenized_log_entries(self, structured_log_file_path):
        raw_log = read_csv(structured_log_file_path)
        result = [structured_log_line[-3].split(' ') for structured_log_line in raw_log[1:]]
        return result
