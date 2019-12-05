from src.data_config import DataConfigs
from data.full.utils import read_template_assignments_from_file
from src.helpers.data_manager import DataManager

DATA_CONFIG = DataConfigs.BGL_FULL
TEMPLATE_ASSIGNMENT_PATH = DATA_CONFIG['assignments_path']
OUTPUT_PATH = 'unstructured/{}_filtered2.log'.format(DATA_CONFIG['name'])

data_manager = DataManager(DATA_CONFIG)
raw_log_entries = data_manager.get_raw_log_entries()
template_assignments = read_template_assignments_from_file(TEMPLATE_ASSIGNMENT_PATH)
input_line_count = len(raw_log_entries)
output_line_count = 0

# Write only lines that have a valid template assignment
with open(OUTPUT_PATH, 'w+', encoding='utf-8') as output_file:
    for idx, tokenized_log_entry in enumerate(raw_log_entries):
        log_entry = ' '.join(tokenized_log_entry)
        if template_assignments[idx] != '-1':
            output_file.write('{}\n'.format(log_entry))
            output_line_count += 1

print('Input File: {} lines'.format(input_line_count))
print('Output File: {} lines ({} filtered)'.format(output_line_count,
                                                   input_line_count - output_line_count))
