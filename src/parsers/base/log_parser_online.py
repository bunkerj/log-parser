from abc import ABC, abstractmethod


class LogParserOnline(ABC):
    @abstractmethod
    def perform_online_em(self, tokenized_log):
        pass

    @abstractmethod
    def get_clusters(self, tokenized_logs):
        pass
