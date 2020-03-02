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
    split_on_result_sources, split_on_samples, get_average_from_samples

N_SAMPLES = 3
LABEL_COUNTS = list(range(0, 601, 100))
N_LABELS = len(LABEL_COUNTS)

data_configs = [
    # DataConfigs.Android,
    DataConfigs.Apache,
    # DataConfigs.BGL,
    # DataConfigs.Hadoop,
    # DataConfigs.HDFS,
    # DataConfigs.HealthApp,
    # DataConfigs.HPC,
    # DataConfigs.Linux,
    # DataConfigs.Mac,
    # DataConfigs.OpenSSH,
    # DataConfigs.OpenStack,
    # DataConfigs.Proxifier,
    # DataConfigs.Spark,
    # DataConfigs.Thunderbird,
    # DataConfigs.Windows,
    # DataConfigs.Zookeeper,
]


def _perform_feedback_experiment(num_label, passed_data_config):
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


if __name__ == '__main__':
    results = {}
    for data_config in data_configs:
        name = data_config['name']
        print('Running for {}...'.format(name))

        start = time()
        with mp.Pool(mp.cpu_count()) as pool:
            total_label_counts = LABEL_COUNTS * N_SAMPLES
            data_config_list = [data_config] * len(total_label_counts)
            arguments = zip(total_label_counts, data_config_list)
            mp_results = pool.starmap(_perform_feedback_experiment, arguments)
        print('Time taken: {}'.format(time() - start))

        lab_impurities, unlab_impurities = split_on_result_sources(mp_results)

        lab_samples = split_on_samples(lab_impurities, N_LABELS)
        unlab_samples = split_on_samples(unlab_impurities, N_LABELS)

        avg_lab_impurities = get_average_from_samples(lab_samples)
        avg_unlab_impurities = get_average_from_samples(unlab_samples)

        tokenized_log_entries = DataManager(data_config) \
            .get_tokenized_no_num_log_entries()

        results[name] = {}
        results[name]['labeled_impurities_samples'] = lab_samples
        results[name]['unlabeled_impurities_samples'] = unlab_samples
        results[name]['avg_labeled_impurities'] = avg_lab_impurities
        results[name]['avg_unlabeled_impurities'] = avg_unlab_impurities
        results[name]['label_counts'] = LABEL_COUNTS
        results[name]['n_logs'] = len(tokenized_log_entries)

        filename = 'feedback_eval_{}_{}s.p'.format(name.lower(), N_SAMPLES)
        dump_results(filename, results)

    print('Done!')
