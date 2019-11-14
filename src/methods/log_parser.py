from src.utils import read_file


class LogParser:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.tokenized_log_entries = self._get_tokenized_log(log_file_path)
        self.unique_event_templates = []
        self.event_template_per_entry = []

    def _get_tokenized_log(self, log_file_path):
        raw_log = read_file(log_file_path)
        log_lines = raw_log.split('\n')
        return [log_line.split(' ') for log_line in log_lines]
