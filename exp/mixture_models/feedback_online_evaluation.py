"""
Evaluate how different initializations have an impact on impurity when using
the online multinomial mixture model.
"""
from time import time
from global_utils import dump_results, get_avg, get_log_labels, \
    get_num_true_clusters
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager
from src.parsers.multinomial_mixture_online import MultinomialMixtureOnline
from global_constants import N_LOGS, LABELED_IMPURITIES_SAMPLES, \
    UNLABELED_IMPURITIES_SAMPLES, LABEL_COUNTS, AVG_LABELED_IMPURITIES, \
    AVG_UNLABELED_IMPURITIES, AVG_LABELED_TIMING, AVG_UNLABELED_TIMING


def run_feedback_online_evaluation(data_config, n_samples, is_class, is_online,
                                   label_counts):
    data_manager = DataManager(data_config)
    tokenized_logs = data_manager.get_tokenized_logs()
    true_assignments = data_manager.get_true_assignments()
    num_true_clusters = get_num_true_clusters(true_assignments)
    ev = Evaluator(true_assignments)

    results = {
        N_LOGS: len(tokenized_logs),
        AVG_LABELED_IMPURITIES: [],
        AVG_UNLABELED_IMPURITIES: [],
        AVG_LABELED_TIMING: 0,
        AVG_UNLABELED_TIMING: 0,
        LABELED_IMPURITIES_SAMPLES: [],
        UNLABELED_IMPURITIES_SAMPLES: [],
        LABEL_COUNTS: label_counts,
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
            lab_parser = MultinomialMixtureOnline(tokenized_logs,
                                                  num_true_clusters,
                                                  is_classification=is_class,
                                                  alpha=1.05,
                                                  beta=1.05)
            unlab_parser = MultinomialMixtureOnline(tokenized_logs,
                                                    num_true_clusters,
                                                    is_classification=is_class,
                                                    alpha=1.05,
                                                    beta=1.05)
            unlab_parser.set_parameters(lab_parser.get_parameters())

            log_labels = get_log_labels(true_assignments, label_count)
            lab_parser.label_logs(log_labels, tokenized_logs)
            labeled_indices = lab_parser.labeled_indices

            lab_time = time()
            if is_online:
                lab_parser.perform_online_batch_em(tokenized_logs)
            else:
                lab_parser.perform_offline_em(tokenized_logs)
            results[AVG_LABELED_TIMING] += (time() - lab_time) / n_samples

            lab_cluster_t = lab_parser.get_clusters(tokenized_logs)
            lab_impurity = ev.get_impurity(lab_cluster_t, labeled_indices)

            unlab_time = time()
            if is_online:
                unlab_parser.perform_online_batch_em(tokenized_logs)
            else:
                unlab_parser.perform_offline_em(tokenized_logs)
            results[AVG_UNLABELED_TIMING] += (time() - unlab_time) / n_samples

            unlab_cluster_t = unlab_parser.get_clusters(tokenized_logs)
            unlab_impurity = ev.get_impurity(unlab_cluster_t, labeled_indices)

            lab_impurities_samples[sample_idx].append(lab_impurity)
            unlab_impurities_samples[sample_idx].append(unlab_impurity)

    print('\nTime taken: {}\n'.format(time() - start))

    results[AVG_LABELED_IMPURITIES] = get_avg(lab_impurities_samples)
    results[AVG_UNLABELED_IMPURITIES] = get_avg(unlab_impurities_samples)
    results[LABELED_IMPURITIES_SAMPLES] = lab_impurities_samples
    results[UNLABELED_IMPURITIES_SAMPLES] = unlab_impurities_samples

    return results


if __name__ == '__main__':
    n_samples = 3
    is_class = False
    is_online = False
    data_config = DataConfigs.Apache
    label_counts = [0, 200, 400, 600, 800, 1000]

    results = run_feedback_online_evaluation(data_config, n_samples, is_class,
                                             is_online, label_counts)
    dump_results('feedback_offline_em_evaluation.p', results)
