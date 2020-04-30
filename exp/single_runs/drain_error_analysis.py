"""
Analyze the errors from Drain for a given dataset.
"""
from global_utils import dump_results
from src.parsers.drain import Drain
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager


def run_drain_error_analysis(data_config, drain_params, name):
    data_manager = DataManager(data_config)
    tokenized_log_entries = data_manager.get_tokenized_log_entries()
    true_assignments = data_manager.get_true_assignments()

    parser = Drain(tokenized_log_entries, *drain_params)
    parser.parse()

    results = {
        'true_assignments': true_assignments,
        'cluster_templates': parser.cluster_templates,
    }
    dump_results(name, results)


if __name__ == '__main__':
    data_config = DataConfigs.Proxifier
    drain_params = (8, 75, 0.4)
    name = 'drain_error_analysis.p'

    run_drain_error_analysis(data_config, drain_params, name)
