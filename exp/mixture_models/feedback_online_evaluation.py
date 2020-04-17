"""
Evaluate how different initializations have an impact on impurity when using
the online multinomial mixture model.
"""
from time import time
from exp.mixture_models.utils import get_log_labels, get_num_true_clusters
from global_utils import dump_results
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager
from src.parsers.multinomial_mixture_online import MultinomialMixtureOnline

N_INIT = 50
N_SAMPLES = 3
DATA_CONFIG = DataConfigs.Apache
LABEL_COUNTS = [0, 200, 400, 600, 800, 1000]

data_manager = DataManager(DATA_CONFIG)
tokenized_log_entries = data_manager.get_tokenized_no_num_log_entries()
true_assignments = data_manager.get_true_assignments()
num_true_clusters = get_num_true_clusters(true_assignments)
evaluator = Evaluator(true_assignments)

results = {
    'n_logs': len(tokenized_log_entries),
    'avg_labeled_impurities': [],
    'avg_unlabeled_impurities': [],
    'labeled_impurities_samples': [],
    'unlabeled_impurities_samples': [],
    'label_counts': LABEL_COUNTS,
}
lab_impurities_samples = []
unlab_impurities_samples = []

start = time()

for sample_idx in range(N_SAMPLES):
    print('Sample {}...'.format(sample_idx))
    lab_impurities_samples.append([])
    unlab_impurities_samples.append([])

    for label_count in LABEL_COUNTS:
        print(label_count)
        lab_parser = MultinomialMixtureOnline(tokenized_log_entries,
                                              num_true_clusters,
                                              is_classification=False,
                                              alpha=1.05,
                                              beta=1.05)
        unlab_parser = MultinomialMixtureOnline(tokenized_log_entries,
                                                num_true_clusters,
                                                is_classification=False,
                                                alpha=1.05,
                                                beta=1.05)
        unlab_parser.set_parameters(lab_parser.get_parameters())

        log_labels = get_log_labels(true_assignments, label_count)
        lab_parser.label_logs(log_labels, tokenized_log_entries)
        labeled_indices = lab_parser.labeled_indices

        lab_parser.perform_online_batch_em(tokenized_log_entries)
        lab_cluster_templates = lab_parser.get_clusters(tokenized_log_entries)
        lab_impurity = evaluator.get_impurity(lab_cluster_templates,
                                              labeled_indices)

        unlab_parser.perform_online_batch_em(tokenized_log_entries)
        unlab_cluster_templates = unlab_parser.get_clusters(
            tokenized_log_entries)
        unlab_impurity = evaluator.get_impurity(unlab_cluster_templates,
                                                labeled_indices)

        lab_impurities_samples[sample_idx].append(lab_impurity)
        unlab_impurities_samples[sample_idx].append(unlab_impurity)

print('\nTime taken: {}\n'.format(time() - start))

for label_idx in range(len(LABEL_COUNTS)):
    avg_lab_impurity = 0
    avg_unlab_impurity = 0
    for sample_idx in range(N_SAMPLES):
        avg_lab_impurity += lab_impurities_samples[sample_idx][label_idx]
        avg_unlab_impurity += unlab_impurities_samples[sample_idx][label_idx]
    results['avg_labeled_impurities'].append(avg_lab_impurity / N_SAMPLES)
    results['avg_unlabeled_impurities'].append(avg_unlab_impurity / N_SAMPLES)

results['labeled_impurities_samples'] = lab_impurities_samples
results['unlabeled_impurities_samples'] = unlab_impurities_samples

result_filename = 'feedback_online_evaluation_{}_{}s.p'.format(
    DATA_CONFIG['name'].lower(), N_SAMPLES)
dump_results(result_filename, results)

print('Done!')
