"""
Evaluate how different initializations have an impact on impurity when using
the online multinomial mixture model. [Using multiprocessing]
"""
from time import time
from copy import deepcopy
import multiprocessing as mp
from global_utils import dump_results
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager
from src.parsers.multinomial_mixture_online import MultinomialMixtureOnline
from exp.mixture_models.utils import get_log_labels, get_num_true_clusters, \
    split_on_result_sources, split_on_samples, get_avg
from global_constants import LABELED_IMPURITIES_SAMPLES, \
    UNLABELED_IMPURITIES_SAMPLES, LABEL_COUNTS, N_LOGS, \
    AVG_LABELED_IMPURITIES, AVG_UNLABELED_IMPURITIES


def perform_single_experiment(num_label, passed_data_config):
    print(passed_data_config['name'])
    data_manager = DataManager(passed_data_config)
    logs = data_manager.get_tokenized_logs()
    true_assignments = data_manager.get_true_assignments()
    num_true_clusters = get_num_true_clusters(true_assignments)
    evaluator = Evaluator(true_assignments)

    lab_parser = MultinomialMixtureOnline(logs,
                                          num_true_clusters,
                                          is_classification=True,
                                          alpha=1.05,
                                          beta=1.05)
    unlab_parser = deepcopy(lab_parser)

    log_labels = get_log_labels(true_assignments, num_label)
    lab_parser.label_logs(log_labels, logs)
    labeled_indices = lab_parser.labeled_indices

    lab_parser.perform_online_batch_em(log_labels)
    unlab_parser.perform_online_batch_em(log_labels)

    lab_clusters = lab_parser.get_clusters(logs)
    unlab_clusters = unlab_parser.get_clusters(logs)

    lab_impurity = evaluator.get_impurity(lab_clusters, labeled_indices)
    unlab_impurity = evaluator.get_impurity(unlab_clusters, labeled_indices)

    return lab_impurity, unlab_impurity


def run_feedback_evaluation_mp(data_configs, n_samples, label_count_values):
    results = {}
    n_labels = len(label_count_values)
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
        tokenized_logs = DataManager(data_config).get_tokenized_logs()

        dataset_results = {
            AVG_LABELED_IMPURITIES: get_avg(lab_samples),
            AVG_UNLABELED_IMPURITIES: get_avg(unlab_samples),
            LABELED_IMPURITIES_SAMPLES: lab_samples,
            UNLABELED_IMPURITIES_SAMPLES: unlab_samples,
            LABEL_COUNTS: label_count_values,
            N_LOGS: len(tokenized_logs)
        }

        results[dataset_name] = dataset_results

    print('Done!')
    return results


if __name__ == '__main__':
    n_samples = 3
    label_count_values = list(range(0, 601, 100))
    data_configs = [
        DataConfigs.Apache,
        DataConfigs.Proxifier,
    ]

    results = run_feedback_evaluation_mp(data_configs, n_samples,
                                         label_count_values)
    dump_results('feedback_evaluation_mp.p', results)
