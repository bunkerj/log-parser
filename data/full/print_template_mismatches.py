from data.full.utils import readlines_with_jump
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager

JUMP_SIZE = 1

data_manager = DataManager(DataConfigs.BGL_FULL)

template_assignments = readlines_with_jump(DataConfigs.BGL_FULL['assignments_path'], JUMP_SIZE)
tokenized_log_entries = data_manager.get_tokenized_log_entries()[::JUMP_SIZE]

for idx in range(len(template_assignments)):
    if template_assignments[idx] == -1:
        print(tokenized_log_entries[idx])

