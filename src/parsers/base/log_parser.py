from abc import ABC, abstractmethod


class LogParser(ABC):
    def __init__(self, tokenized_logs):
        self.tokenized_logs = tokenized_logs
        self.cluster_templates = {}

    @abstractmethod
    def parse(self):
        pass
