import os
from os.path import expanduser
from data.scripts.utils import get_nested_file_paths

DATASET_NAME = 'OpenStack'
INPUT_DIR = expanduser('~/Desktop/log_datasets/{}/'.format(DATASET_NAME))
OUTPUT_DIR = expanduser('~/Desktop/processed_dataset_logs/')
OUTPUT_FILE = expanduser(OUTPUT_DIR + '{}.log'.format(DATASET_NAME.lower()))

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

input_file_paths = get_nested_file_paths(INPUT_DIR, '*.log')
n_input_files = len(input_file_paths)

with open(OUTPUT_FILE, 'w+', encoding='utf-8') as output_file:
    for idx, input_file_path in enumerate(input_file_paths, start=1):
        print('File {}/{}...'.format(idx, n_input_files))
        with open(input_file_path, 'r+', encoding='utf-8') as input_file:
            for line in input_file:
                output_file.write(line)

print("Done!")
