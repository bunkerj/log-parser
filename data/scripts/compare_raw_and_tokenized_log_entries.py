"""
Compare the raw and tokenized logs for some specified indices.
"""
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager

DATA_CONFIG = DataConfigs.HPC
INDICES = (401, 562, 699)

data_manager = DataManager(DATA_CONFIG)
data_manager.print_select_raw_and_tokenized_logs(INDICES)
