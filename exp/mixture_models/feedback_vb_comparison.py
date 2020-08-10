"""
Compare MultinomialMixtureVB scores with and without pairwise constraints for
different datasets.
"""
from copy import deepcopy
from global_utils import dump_results
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager
from src.helpers.evaluator import Evaluator
from exp.mixture_models.utils import get_num_true_clusters
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


def run_drain_performance_nmi(n_samples, data_config):
    data_manager = DataManager(data_config)
    tokenized_logs = data_manager.get_tokenized_logs()
    true_assignments = data_manager.get_true_assignments()
    num_clusters = get_num_true_clusters(true_assignments)
    evaluator = Evaluator(true_assignments)
    oracle = Oracle(true_assignments)

    reg_scores = []
    feedback_scores = []

    for _ in range(n_samples):
        parser_reg = MultinomialMixtureVB(tokenized_logs, num_clusters)
        parser_reg.parse()
        parser_feedback = deepcopy(parser_reg)

        W = oracle.get_constraints_matrix(
            parsed_clusters=parser_reg.cluster_templates,
            n_constraint_samples=10,
            tokenized_logs=tokenized_logs,
            weight=1000)

        parser_feedback.provide_constraints(W)
        parser_feedback.parse()

        acc_reg = evaluator.get_accuracy(parser_reg.cluster_templates)
        acc_feedback = evaluator.get_accuracy(parser_feedback.cluster_templates)

        reg_scores.append(acc_reg)
        feedback_scores.append(acc_feedback)

    return {
        'regular': reg_scores,
        'feedback': feedback_scores,
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

    n_samples = 5

    for data_config in data_configs:
        name = data_config['name'].lower()
        print(name)

        results = run_drain_performance_nmi(n_samples, data_config)
        filename = 'feedback_vb_comparison_{}.p'.format(name)

        dump_results(filename, results)
