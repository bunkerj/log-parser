"""
Print and save the accuracies of a single Drain run on a target dataset
(DATA_CONFIG).
"""
from time import time
from global_utils import dump_results
from src.parsers.enhanced_drain import EnhancedDrain
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator
from src.helpers.data_manager import DataManager


def run_drain_exp(data_config, enhanced_drain_params, name):
    data_manager = DataManager(data_config)
    tokenized_log_entries = data_manager.get_tokenized_log_entries()
    true_assignments = data_manager.get_true_assignments()
    parser = EnhancedDrain(tokenized_log_entries, *enhanced_drain_params)

    start_time = time()
    parser.parse()
    minutes_to_parse = (time() - start_time) / 60

    start_time = time()
    evaluator = Evaluator(true_assignments)
    accuracy = evaluator.evaluate(parser.cluster_templates)
    minutes_to_eval = (time() - start_time) / 60

    results = {
        'accuracy': accuracy,
        'mins_to_parse': minutes_to_parse,
        'mins_to_eval': minutes_to_eval,
        'parser': parser,
    }

    dump_results(name, results)


if __name__ == '__main__':
    data_config = DataConfigs.Proxifier
    enhanced_drain_params = (8.39, 34, 0.24, 0.18)
    name = 'enhanced_drain_exp.p'

    run_drain_exp(data_config, enhanced_drain_params, name)
