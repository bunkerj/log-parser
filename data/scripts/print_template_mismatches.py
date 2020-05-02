"""
Prints all of the templates of a given dataset that are unmatched (i.e has a
template index of -1).
"""
from data.scripts.utils import read_template_assignments_from_file
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager

JUMP_SIZE = 1
DATA_CONFIG = DataConfigs.BGL_FULL

data_manager = DataManager(DATA_CONFIG)
assignment_path = DATA_CONFIG['assignments_path']

template_assignments = read_template_assignments_from_file(assignment_path,
                                                           JUMP_SIZE)
tokenized_log_entries = data_manager.get_tokenized_logs()[::JUMP_SIZE]

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
