from abc import ABC, abstractmethod


class LogParserOnline(ABC):
    @abstractmethod
    def process_single_log(self, tokenized_log):
        pass

    @abstractmethod
    def get_clusters(self, tokenized_log_entries):
        pass
