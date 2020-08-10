"""
Compare MultinomialMixtureVB scores with and without pairwise constraints for
different datasets.
"""
from copy import deepcopy
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


def run_feedback_vb_comparison(n_label, n_constraints, n_samples, data_config):
    data_manager = DataManager(data_config)
    tokenized_logs = data_manager.get_tokenized_logs()
    true_assignments = data_manager.get_true_assignments()
    num_clusters = get_num_true_clusters(true_assignments)
    evaluator = Evaluator(true_assignments)
    oracle = Oracle(true_assignments)

    scores_base = []
    scores_lab = []
    scores_lab_const = []

    for _ in range(n_samples):
        parser = MultinomialMixtureVB(tokenized_logs, num_clusters)

        parser_lab = deepcopy(parser)
        log_labels = get_log_labels(true_assignments, n_label)
        parser_lab.label_logs(log_labels)
        labeled_indices = parser.labeled_indices

        parser_lab_const = deepcopy(parser_lab)
        W = oracle.get_constraints_matrix(
            parsed_clusters=parser.cluster_templates,
            n_constraint_samples=n_constraints,
            tokenized_logs=tokenized_logs,
            weight=1000)
        parser_lab_const.provide_constraints(W)

        parser.parse()
        parser_lab.parse()
        parser_lab_const.parse()

        c = parser.cluster_templates
        c_lab = parser_lab.cluster_templates
        c_lab_const = parser_lab_const.cluster_templates

        score = evaluator.get_impurity(c, labeled_indices)
        score_lab = evaluator.get_impurity(c_lab, labeled_indices)
        score_lab_const = evaluator.get_impurity(c_lab_const, labeled_indices)

        scores_base.append(score)
        scores_lab.append(score_lab)
        scores_lab_const.append(score_lab_const)

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
        results = run_feedback_vb_comparison(n_label=20,
                                             n_constraints=50,
                                             n_samples=50,
                                             data_config=data_config)
        filename = 'feedback_vb_comparison_{}.p'.format(name)
        dump_results(filename, results)
