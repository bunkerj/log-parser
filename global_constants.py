PLACEHOLDER = '<*>'
RESULTS_DIR = 'results'
# SPLIT_REGEX = r'[\s=:,]'  # For IPLoM consistency
SPLIT_REGEX = r'[\s]'  # For Drain consistency

MAX_NEG_VALUE = -99999999999
ZERO_THRESHOLD = 0.0000001

# For experiments pipeline
NAME = '__name__'
FUNCTION = '__function__'


class MAP:
    ONE_TO_ONE = 'ONE_TO_ONE'
    ONE_TO_MANY = 'ONE_TO_MANY'
    MANY_TO_ONE = 'MANY_TO_ONE'
    MANY_TO_MANY = 'MANY_TO_MANY'


CANNOT_LINK = 'CANNOT_LINK'
MUST_LINK = 'MUST_LINK'

# Mixture experiment results constants
N_LOGS = 'n_logs'
LABEL_COUNTS = 'label_counts'
AVG_LABELED_TIMING = 'avg_labeled_timing'
AVG_UNLABELED_TIMING = 'avg_unlabeled_timing'
AVG_LABELED_IMPURITIES = 'avg_labeled_impurities'
AVG_UNLABELED_IMPURITIES = 'avg_unlabeled_impurities'
LABELED_IMPURITIES_SAMPLES = 'labeled_impurities_samples'
UNLABELED_IMPURITIES_SAMPLES = 'unlabeled_impurities_samples'
