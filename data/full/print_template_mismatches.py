from data.full.utils import readlines_with_jump
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager

JUMP_SIZE = 1

data_manager = DataManager(DataConfigs.BGL_FULL)

template_assignments = readlines_with_jump(DataConfigs.BGL_FULL['assignments_path'], JUMP_SIZE)
tokenized_log_entries = data_manager.get_tokenized_log_entries()[::JUMP_SIZE]

n = len(template_assignments)

# Get template mismatches
mismatches = set()
mismatch_count = 0
for idx in range(n):
    log_entry = ' '.join(tokenized_log_entries[idx])
    if template_assignments[idx] == '-1':
        mismatches.add(log_entry)
        mismatch_count += 1

# Print template mismatches
for mismatch in mismatches:
    print(mismatch)

print('\n-------------------------------------\n')
print('Number of mismatches: {}'.format(mismatch_count))
print('Percentage matched: {}'.format(100 * (n - mismatch_count) / n))
