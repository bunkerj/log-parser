"""
Evaluate how different initializations have an impact on impurity when using
the multinomial mixture model. [Using multiprocessing]
"""
import multiprocessing as mp
from time import time
from exp.utils import dump_results
from src.parsers.multinomial_mixture import MultinomialMixture
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager
from src.utils import get_template_assignments
from exp.mixture_models.utils import get_log_labels, get_num_true_clusters, \
    split_on_result_sources, split_on_samples, get_average_from_samples

N_SAMPLES = 10
DATA_CONFIG = DataConfigs.Apache
LABEL_COUNTS = list(range(0, 401, 25))
N_LABELS = len(LABEL_COUNTS)

data_manager = DataManager(DATA_CONFIG)
tokenized_log_entries = data_manager.get_tokenized_no_num_log_entries()
true_assignments = get_template_assignments(DATA_CONFIG['assignments_path'])
num_true_clusters = get_num_true_clusters(true_assignments)
evaluator = Evaluator(true_assignments)

N_logs = len(tokenized_log_entries)

results = {
    'labeled_impurities': [],
    'unlabeled_impurities': [],
    'label_counts': LABEL_COUNTS,
    'n_logs': len(tokenized_log_entries),
}


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

    lab_impurities, unlab_impurities = split_on_result_sources(mp_results)

    lab_samples = split_on_samples(lab_impurities, N_LABELS)
    unlab_samples = split_on_samples(unlab_impurities, N_LABELS)

    avg_lab_impurities = get_average_from_samples(lab_samples)
    avg_unlab_impurities = get_average_from_samples(unlab_samples)

    results['labeled_impurities_samples'] = lab_samples
    results['unlabeled_impurities_samples'] = unlab_samples
    results['avg_labeled_impurities'] = avg_lab_impurities
    results['avg_unlabeled_impurities'] = avg_unlab_impurities

    print('Time taken: {}'.format(time() - start))

    result_filename = 'feedback_evaluation_mp_filtered_no_num_{}_{}s.p'.format(
        DATA_CONFIG['name'].lower(), N_SAMPLES)
    dump_results(result_filename, results)

    print('Done!')
