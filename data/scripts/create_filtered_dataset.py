"""
Create a new dataset from a base dataset (DATA_CONFIG) and its corresponding
template assignment file (TEMPLATE_ASSIGNMENT_PATH).

The new dataset is created by filtering the base dataset lines that do not have
a template assignment (i.e has a template index of -1).
"""
from src.data_config import DataConfigs
from data.scripts.utils import read_template_assignments_from_file
from src.helpers.data_manager import DataManager

DATA_CONFIG = DataConfigs.BGL_FULL
TEMPLATE_ASSIGNMENT_PATH = DATA_CONFIG['assignments_path']
OUTPUT_PATH = 'data/full/unstructured/{}_filtered.log'.format(
    DATA_CONFIG['name'])

data_manager = DataManager(DATA_CONFIG)
raw_log_full_lines = data_manager.get_raw_log_full_lines()
template_assignments = read_template_assignments_from_file(
    TEMPLATE_ASSIGNMENT_PATH)
input_line_count = len(raw_log_full_lines)
output_line_count = 0

# Only write lines that have a valid template assignment
with open(OUTPUT_PATH, 'w+', encoding='utf-8') as output_file:
    for idx, tokenized_log_entry in enumerate(raw_log_full_lines):
        log_entry = ' '.join(tokenized_log_entry)
        if template_assignments[idx] != '-1':
            output_file.write('{}\n'.format(log_entry))
            output_line_count += 1

print('Input File: {} lines'.format(input_line_count))
print('Output File: {} lines ({} filtered)'.format(output_line_count,
                                                   input_line_count
                                                   - output_line_count))
