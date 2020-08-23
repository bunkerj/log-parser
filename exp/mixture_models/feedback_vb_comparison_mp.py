"""
Compare MultinomialMixtureVB scores with and without pairwise constraints for
different datasets.
"""
import multiprocessing as mp
from global_utils import dump_results
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.helpers.evaluator import Evaluator
from exp.mixture_models.utils import get_num_true_clusters, get_log_labels
from src.parsers.multinomial_mixture_vb import MultinomialMixtureVB
from src.helpers.oracle import Oracle


def get_mappings(cluster_templates, tokenized_logs):
    new_cluster_templates = {}
    for cluster in cluster_templates:
        new_cluster_templates[cluster] = []
        for idx in cluster_templates[cluster]:
            token_str = ' '.join(tokenized_logs[idx])
            new_cluster_templates[cluster].append(token_str)
    return new_cluster_templates


def run_single_sample_exp(logs, true_assignments, num_clusters, n_label,
                          evaluator, oracle, n_constraints):
    log_labels = get_log_labels(true_assignments, n_label)

    # Baseline
    parser = MultinomialMixtureVB()
    parser.fit(logs, num_clusters)
    c = parser.predict(logs)

    # Labeled Model
    parser_lab = MultinomialMixtureVB()
    parser_lab.fit(logs, num_clusters, log_labels=log_labels)
    labeled_indices = parser_lab.get_labeled_indices()
    c_lab = parser_lab.predict(logs)

    # Labeled + Constrained Model
    parser_lab_const = MultinomialMixtureVB()
    W = oracle.get_constraints_matrix(
        parsed_clusters=c_lab,
        n_constraint_samples=n_constraints,
        tokenized_logs=logs,
        weight=1)
    parser_lab_const.fit(logs, num_clusters, log_labels=log_labels,
                         constraints=W)
    c_lab_const = parser_lab_const.predict(logs)

    # Calculate Scores
    score = evaluator.get_impurity(c, labeled_indices)
    score_lab = evaluator.get_impurity(c_lab, labeled_indices)
    score_lab_const = evaluator.get_impurity(c_lab_const, labeled_indices)

    return score, score_lab, score_lab_const


def run_feedback_vb_comparison(n_label, n_constraints, n_samples, data_config):
    data_manager = DataManager(data_config)
    logs = data_manager.get_tokenized_logs()
    true_assignments = data_manager.get_true_assignments()
    num_clusters = get_num_true_clusters(true_assignments)
    evaluator = Evaluator(true_assignments)
    oracle = Oracle(true_assignments)

    with mp.Pool(mp.cpu_count()) as pool:
        arguments = [(logs, true_assignments, num_clusters, n_label,
                      evaluator, oracle, n_constraints) for _ in
                     range(n_samples)]
        mp_results = pool.starmap(run_single_sample_exp, arguments)
        scores_base, scores_lab, scores_lab_const \
            = list(zip(*mp_results))

    return {
        'base': scores_base,
        'labeled': scores_lab,
        'labeled_const': scores_lab_const,
    }


if __name__ == '__main__':
    data_configs = [
        DataConfigs.Android,
        DataConfigs.Apache,
        DataConfigs.BGL,
        DataConfigs.Hadoop,
        DataConfigs.HDFS,
        DataConfigs.HealthApp,
        DataConfigs.HPC,
        DataConfigs.Linux,
    ]

    for data_config in data_configs:
        name = data_config['name'].lower()
        print(name)
        results = run_feedback_vb_comparison(n_label=50,
                                             n_constraints=100,
                                             n_samples=1000,
                                             data_config=data_config)
        filename = 'feedback_vb_comparison_{}.p'.format(name)
        dump_results(filename, results)
