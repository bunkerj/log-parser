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


def run_feedback_online_evaluation(n_samples, is_class, is_online,
                                   data_config, label_counts, name):
    data_manager = DataManager(data_config)
    tokenized_log_entries = data_manager.get_tokenized_no_num_log_entries()
    true_assignments = data_manager.get_true_assignments()
    num_true_clusters = get_num_true_clusters(true_assignments)
    ev = Evaluator(true_assignments)

    results = {
        'n_logs': len(tokenized_log_entries),
        'avg_labeled_impurities': [],
        'avg_unlabeled_impurities': [],
        'avg_labeled_timing': 0,
        'avg_unlabeled_timing': 0,
        'labeled_impurities_samples': [],
        'unlabeled_impurities_samples': [],
        'label_counts': label_counts,
    }
    lab_impurities_samples = []
    unlab_impurities_samples = []

    print(data_config['name'])

    start = time()

    for sample_idx in range(n_samples):
        print('Sample {}...'.format(sample_idx))
        lab_impurities_samples.append([])
        unlab_impurities_samples.append([])

        for label_count in label_counts:
            print(label_count)
            lab_parser = MultinomialMixtureOnline(tokenized_log_entries,
                                                  num_true_clusters,
                                                  is_classification=is_class,
                                                  alpha=1.05,
                                                  beta=1.05)
            unlab_parser = MultinomialMixtureOnline(tokenized_log_entries,
                                                    num_true_clusters,
                                                    is_classification=is_class,
                                                    alpha=1.05,
                                                    beta=1.05)
            unlab_parser.set_parameters(lab_parser.get_parameters())

            log_labels = get_log_labels(true_assignments, label_count)
            lab_parser.label_logs(log_labels, tokenized_log_entries)
            labeled_indices = lab_parser.labeled_indices

            lab_time = time()
            if is_online:
                lab_parser.perform_online_batch_em(tokenized_log_entries)
            else:
                lab_parser.perform_offline_em(tokenized_log_entries)
            results['avg_labeled_timing'] += (time() - lab_time) / n_samples

            lab_cluster_t = lab_parser.get_clusters(tokenized_log_entries)
            lab_impurity = ev.get_impurity(lab_cluster_t, labeled_indices)

            unlab_time = time()
            if is_online:
                unlab_parser.perform_online_batch_em(tokenized_log_entries)
            else:
                unlab_parser.perform_offline_em(tokenized_log_entries)
            results['avg_unlabeled_timing'] += (time() - unlab_time) / n_samples

            unlab_cluster_t = unlab_parser.get_clusters(tokenized_log_entries)
            unlab_impurity = ev.get_impurity(unlab_cluster_t, labeled_indices)

            lab_impurities_samples[sample_idx].append(lab_impurity)
            unlab_impurities_samples[sample_idx].append(unlab_impurity)

    print('\nTime taken: {}\n'.format(time() - start))

    for label_idx in range(len(label_counts)):
        avg_lab_impurity = 0
        avg_unlab_impurity = 0
        for sample_idx in range(n_samples):
            avg_lab_impurity += lab_impurities_samples[sample_idx][label_idx]
            avg_unlab_impurity += unlab_impurities_samples[sample_idx][
                label_idx]
        results['avg_labeled_impurities'].append(avg_lab_impurity / n_samples)
        results['avg_unlabeled_impurities'].append(
            avg_unlab_impurity / n_samples)

    results['labeled_impurities_samples'] = lab_impurities_samples
    results['unlabeled_impurities_samples'] = unlab_impurities_samples

    dump_results(name, results)


if __name__ == '__main__':
    n_samples = 3
    is_class = False
    is_online = False
    data_config = DataConfigs.Apache
    label_counts = [0, 200, 400, 600, 800, 1000]
    name = 'feedback_offline_em_evaluation.p'

    run_feedback_online_evaluation(n_samples, is_class, is_online,
                                   data_config, label_counts, name)
