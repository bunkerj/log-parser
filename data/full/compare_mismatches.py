from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from data.full.utils import read_template_assignments_from_file

PATH = 'assignments/BGL_assignments_reduced_regex.csv'
REF_PATH = 'assignments/BGL_assignments.csv'

template_assignments = read_template_assignments_from_file(PATH)
template_assignments_ref = read_template_assignments_from_file(REF_PATH)

data_manager = DataManager(DataConfigs.BGL_FULL)
tokenized_log_entries = data_manager.get_tokenized_log_entries()

total_mismatches = 0
mismatch_ref_template_indices = set()
for idx, template_idx in enumerate(template_assignments):
    ref_template_idx = template_assignments_ref[idx]
    if template_assignments[idx] == '-1' and ref_template_idx != '-1':
        total_mismatches += 1
        mismatch_token = ' '.join(tokenized_log_entries[idx])
        if ref_template_idx not in mismatch_ref_template_indices:
            mismatch_ref_template_indices.add(ref_template_idx)
            print('[{}]\t{}'.format(ref_template_idx, mismatch_token))

print('\nTotal mismatches: {}'.format(total_mismatches))