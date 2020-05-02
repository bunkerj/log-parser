"""
Plot the accuracy over the maximum tree depth.
"""
from global_utils import dump_results
from src.parsers.drain import Drain
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager


def run_drain_accuracy_over_tree_depth(data_config, tree_depths):
    accuracies = []
    data_manager = DataManager(data_config)
    tokenized_log_entries = data_manager.get_tokenized_logs()
    true_assignments = data_manager.get_true_assignments()
    evaluator = Evaluator(true_assignments)

    for tree_depth in tree_depths:
        parser = Drain(tokenized_log_entries, 3, tree_depth, 0.5)
        parser.parse()
        accuracies.append(evaluator.evaluate(parser.cluster_templates))

    return {
        'tree_depths': tree_depths,
        'accuracies': accuracies,
    }


if __name__ == '__main__':
    data_config = DataConfigs.BGL
    tree_depths = list(range(3, 31, 1))

    results = run_drain_accuracy_over_tree_depth(data_config, tree_depths)
    dump_results('drain_accuracy_over_tree_depth.p', results)
