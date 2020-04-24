from abc import ABC, abstractmethod


class LogParser(ABC):
    def __init__(self, tokenized_log_entries):
        self.tokenized_log_entries = tokenized_log_entries
        self.cluster_templates = {}

    @abstractmethod
    def parse(self):
        pass
