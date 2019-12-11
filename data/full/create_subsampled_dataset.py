"""
Create a new dataset and template assignment files that contains N_SAMPLES
number of samples from the original dataset and template assignment files.
"""
from numpy.random import choice
from data.full.utils import subsample_file, get_file_length

N_SAMPLES = 50000

INPUT_DATASET_PATH = 'data/full/unstructured/BGL_filtered.log'
INPUT_TEMPLATE_ASSIGNMENT_PATH = \
    'data/full/assignments/BGL_filtered_assignments.csv'

OUTPUT_DATASET_PATH = 'data/full/unstructured/BGL_filtered_reduced.log'
OUTPUT_TEMPLATE_ASSIGNMENT_PATH = \
    'data/full/assignments/BGL_filtered_reduced_assignments.csv'

n = get_file_length(INPUT_DATASET_PATH)
m = get_file_length(INPUT_TEMPLATE_ASSIGNMENT_PATH) - 1

if n != m:
    raise Exception('File length mismatch: {} != {}'.format(n, m))
elif n < N_SAMPLES:
    raise Exception(
        'Sample size {} is greater than number of input lines {}'.format(
            N_SAMPLES, n))

indices = set(choice(n, N_SAMPLES, replace=False))

subsample_file(INPUT_DATASET_PATH,
               OUTPUT_DATASET_PATH,
               indices)
subsample_file(INPUT_TEMPLATE_ASSIGNMENT_PATH,
               OUTPUT_TEMPLATE_ASSIGNMENT_PATH,
               indices, include_header=True)
