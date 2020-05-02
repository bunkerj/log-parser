import ntpath
from os.path import expanduser
from data.scripts.utils import get_nested_file_paths, get_num_lines

INPUT_DIR = expanduser('~/Desktop/processed_dataset_logs/')
EXT = '*.log'

for input_file in get_nested_file_paths(INPUT_DIR, EXT):
    n_lines = get_num_lines(input_file)
    basename = ntpath.basename(input_file)
    print('{}: {}'.format(basename, n_lines))
