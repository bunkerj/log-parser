"""
Evaluate how different initializations have an impact on impurity when using
the multinomial mixture model. [Using multiprocessing]
"""
import multiprocessing as mp
from time import time
from exp.mixture_models.utils import get_log_labels, get_num_true_clusters
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
num_true_clusters = get_num_true_clusters(true_assignments)
evaluator = Evaluator(true_assignments)

results = {
    'labeled_impurities': [],
    'unlabeled_impurities': [],
    'label_counts': LABEL_COUNTS,
}


def get_impurity_results_from_jobs(results, n_label_counts, n_samples):
    avg_lab_impurities = []
    avg_unlab_impurities = []
    for label_count_idx in range(n_label_counts):
        avg_lab_impurities.append(0)
        avg_unlab_impurities.append(0)
        for value in results[label_count_idx::n_label_counts]:
            lab_impurity, unlab_impurity = value
            avg_lab_impurities[label_count_idx] += lab_impurity / n_samples
            avg_unlab_impurities[label_count_idx] += unlab_impurity / n_samples
    return avg_lab_impurities, avg_unlab_impurities


def _perform_feedback_experiment(num_label):
    lab_parser = MultinomialMixture(tokenized_log_entries,
                                    num_true_clusters)
    unlab_parser = MultinomialMixture(tokenized_log_entries,
                                      num_true_clusters)
    unlab_parser.initialize_responsibilities(lab_parser)
    log_labels = get_log_labels(true_assignments, num_label)
    lab_parser.label_logs(log_labels)
    labeled_indices = lab_parser.labeled_indices

    lab_parser.parse()
    lab_impurity = evaluator.get_impurity(lab_parser.cluster_templates,
                                          labeled_indices)
    unlab_parser.parse()
    unlab_impurity = evaluator.get_impurity(unlab_parser.cluster_templates,
                                            labeled_indices)
    return lab_impurity, unlab_impurity


if __name__ == '__main__':
    start = time()

    with mp.Pool(mp.cpu_count()) as pool:
        total_label_counts = LABEL_COUNTS * N_SAMPLES
        mp_results = pool.map(_perform_feedback_experiment, total_label_counts)

    avg_lab_impurities, avg_unlab_impurities = \
        get_impurity_results_from_jobs(mp_results, len(LABEL_COUNTS), N_SAMPLES)

    results['labeled_impurities'] = avg_lab_impurities
    results['unlabeled_impurities'] = avg_unlab_impurities

    print('Time taken: {}'.format(time() - start))

    result_filename = 'feedback_evaluation_mp_{}_{}s.p'.format(
        DATA_CONFIG['name'].lower(), N_SAMPLES)
    dump_results(result_filename, results)

    print('Done!')
