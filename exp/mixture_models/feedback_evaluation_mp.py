"""
Evaluate how different initializations have an impact on impurity when using
the multinomial mixture model. [Using multiprocessing]
"""
import multiprocessing as mp
from time import time
from global_utils import dump_results
from src.parsers.multinomial_mixture import MultinomialMixture
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager
from exp.mixture_models.utils import get_log_labels, get_num_true_clusters, \
    split_on_result_sources, split_on_samples, get_avg
from global_constants import LABELED_IMPURITIES_SAMPLES, \
    UNLABELED_IMPURITIES_SAMPLES, LABEL_COUNTS, N_LOGS, \
    AVG_LABELED_IMPURITIES, AVG_UNLABELED_IMPURITIES


def perform_single_experiment(num_label, passed_data_config):
    print(passed_data_config['name'])
    data_manager = DataManager(passed_data_config)
    log_entries = data_manager.get_tokenized_no_num_log_entries()
    true_assignments = data_manager.get_true_assignments()
    num_true_clusters = get_num_true_clusters(true_assignments)
    evaluator = Evaluator(true_assignments)

    lab_parser = MultinomialMixture(log_entries, num_true_clusters)
    unlab_parser = MultinomialMixture(log_entries, num_true_clusters)
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


def run_feedback_evaluation_mp(n_samples, label_count_values, n_labels, name):
    results = {}
    for data_config in data_configs:
        dataset_name = data_config['name']
        print('Running for {}...'.format(dataset_name))

        start = time()
        with mp.Pool(mp.cpu_count()) as pool:
            total_label_counts = label_count_values * n_samples
            data_config_list = [data_config] * len(total_label_counts)
            arguments = zip(total_label_counts, data_config_list)
            mp_results = pool.starmap(perform_single_experiment, arguments)
        print('Time taken: {}'.format(time() - start))

        lab_impurities, unlab_impurities = split_on_result_sources(mp_results)

        lab_samples = split_on_samples(lab_impurities, n_labels)
        unlab_samples = split_on_samples(unlab_impurities, n_labels)

        tokenized_log_entries = DataManager(data_config) \
            .get_tokenized_no_num_log_entries()

        dataset_results = {
            AVG_LABELED_IMPURITIES: get_avg(lab_samples),
            AVG_UNLABELED_IMPURITIES: get_avg(unlab_samples),
            LABELED_IMPURITIES_SAMPLES: lab_samples,
            UNLABELED_IMPURITIES_SAMPLES: unlab_samples,
            LABEL_COUNTS: label_count_values,
            N_LOGS: len(tokenized_log_entries)
        }

        results[dataset_name] = dataset_results

    dump_results(name, results)

    print('Done!')


if __name__ == '__main__':
    n_samples = 3
    label_count_values = list(range(0, 601, 100))
    n_labels = len(label_count_values)
    data_configs = [
        DataConfigs.Apache,
        DataConfigs.Proxifier,
    ]
    name = 'feedback_evaluation_mp.p'

    run_feedback_evaluation_mp(n_samples, label_count_values, n_labels, name)
