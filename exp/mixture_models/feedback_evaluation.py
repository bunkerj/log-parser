"""
Evaluate how different initializations have an impact on impurity when using
the multinomial mixture model.
"""
from time import time
from exp.mixture_models.utils import get_log_labels
from exp.utils import dump_results
from src.parsers.multinomial_mixture import MultinomialMixture
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager
from src.utils import get_template_assignments

N_SAMPLES = 5
DATA_CONFIG = DataConfigs.Apache
LABEL_COUNTS = [0, 200, 400, 600, 800, 1000]

data_manager = DataManager(DATA_CONFIG)
tokenized_log_entries = data_manager.get_tokenized_log_entries()
true_assignments = get_template_assignments(DATA_CONFIG['assignments_path'])
num_true_clusters = len(set(log_data[-1] for log_data in true_assignments))
evaluator = Evaluator(true_assignments)

results = {
    'labeled_impurities': [],
    'unlabeled_impurities': [],
    'label_counts': LABEL_COUNTS,
}

start = time()

for num_label in LABEL_COUNTS:
    print('Processing for {}...'.format(num_label))
    lab_impurity = 0
    unlab_impurity = 0
    for i in range(N_SAMPLES):
        lab_parser = MultinomialMixture(tokenized_log_entries,
                                        num_true_clusters)
        unlab_parser = MultinomialMixture(tokenized_log_entries,
                                          num_true_clusters)
        unlab_parser.initialize_responsibilities(lab_parser)

        log_labels = get_log_labels(true_assignments, num_label)
        lab_parser.label_logs(log_labels)
        labeled_indices = lab_parser.labeled_indices

        lab_parser.parse()
        lab_impurity += evaluator.get_impurity(lab_parser.cluster_templates,
                                               labeled_indices) / N_SAMPLES
        unlab_parser.parse()
        unlab_impurity += evaluator.get_impurity(unlab_parser.cluster_templates,
                                                 labeled_indices) / N_SAMPLES

    results['labeled_impurities'].append(lab_impurity)
    results['unlabeled_impurities'].append(unlab_impurity)

print('\nTime taken: {}\n'.format(time() - start))

result_filename = 'feedback_evaluation_{}_{}s.p'.format(
    DATA_CONFIG['name'].lower(), N_SAMPLES)
dump_results(result_filename, results)

print('Done!')
