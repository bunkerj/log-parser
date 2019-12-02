from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager

JUMP_SIZE = 1000


def readlines_with_jump(file_path, jump_size):
    lines = []
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            if idx % jump_size == 0:
                lines.append(line.strip())
    return lines


data_manager = DataManager(DataConfigs.BGL_FULL)

template_assignments = readlines_with_jump(DataConfigs.BGL_FULL['assignments_path'], JUMP_SIZE)
templates = data_manager.get_templates()
tokenized_log_entries = data_manager.get_tokenized_log_entries()[::JUMP_SIZE]

for idx in range(len(template_assignments)):
    if template_assignments[idx] == -1:
        continue
    template = list(filter(lambda t: t.idx == template_assignments[idx], templates))[0]
    template_str = ' '.join(template.tokens)
    log_entry_str = ' '.join(tokenized_log_entries[idx])
    print('{} |<------------->| {}'.format(template_str, log_entry_str))
