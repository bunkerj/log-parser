"""
Print template assignments of a given dataset configuration. The specified dataset is subsampled by the increment
specified by JUMP_SIZE.
"""
from data.full.utils import read_template_assignments_from_file
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager

JUMP_SIZE = 1000
DATA_CONFIG = DataConfigs.BGL_FULL

data_manager = DataManager(DATA_CONFIG)
templates = data_manager.get_templates()
tokenized_log_entries = data_manager.get_tokenized_log_entries()[::JUMP_SIZE]
assignments = read_template_assignments_from_file(DATA_CONFIG['assignments_path'], JUMP_SIZE)

# Get template mappings
template_mappings = {}
for idx in range(len(assignments)):
    if assignments[idx] == '-1':
        continue
    template = list(filter(lambda t: t.idx == assignments[idx], templates))[0]
    template_str = ' '.join(template.tokens)
    log_entry_str = ' '.join(tokenized_log_entries[idx])
    if template_str not in template_mappings:
        template_mappings[template_str] = set()
    template_mappings[template_str].add(log_entry_str)

# Print template mappings
for template_str in template_mappings:
    print(template_str)
    for log_entry_str in template_mappings[template_str]:
        print(log_entry_str)
    print()
