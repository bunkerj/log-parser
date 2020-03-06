PLACEHOLDER = '<*>'
RESULTS_DIR = 'results'
# SPLIT_REGEX = r'[\s=:,]'  # For IPLoM consistency
SPLIT_REGEX = r'[\s]'  # For Drain consistency


class MAP:
    ONE_TO_ONE = 'ONE_TO_ONE'
    ONE_TO_MANY = 'ONE_TO_MANY'
    MANY_TO_ONE = 'MANY_TO_ONE'
    MANY_TO_MANY = 'MANY_TO_MANY'


# Mixture experiment results constants
LABELED_IMPURITIES_SAMPLES = 'labeled_impurities_samples'
UNLABELED_IMPURITIES_SAMPLES = 'unlabeled_impurities_samples'
AVG_LABELED_IMPURITIES = 'avg_labeled_impurities'
AVG_UNLABELED_IMPURITIES = 'avg_unlabeled_impurities'
LABEL_COUNTS = 'label_counts'
N_LOGS = 'n_logs'
