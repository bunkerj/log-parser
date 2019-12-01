from src.helpers.data_manager import DataManager
from abc import ABC, abstractmethod

SPLIT_REGEX = r'[\s=:,]'


class LogParser(ABC):
    def __init__(self, data_config):
        data_manager = DataManager(data_config)
        self.tokenized_log_entries = data_manager.get_tokenized_log_entries()
        self.cluster_templates = {}

    @abstractmethod
    def parse(self):
        pass
