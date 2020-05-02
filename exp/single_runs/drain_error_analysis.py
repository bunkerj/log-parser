"""
Analyze the errors from Drain for a given dataset.
"""
from global_utils import dump_results
from src.parsers.drain import Drain
from src.data_config import DataConfigs
from src.helpers.data_manager import DataManager


def run_drain_error_analysis(data_config, drain_params):
    data_manager = DataManager(data_config)
    tokenized_log_entries = data_manager.get_tokenized_logs()
    true_assignments = data_manager.get_true_assignments()

    parser = Drain(tokenized_log_entries, *drain_params)
    parser.parse()

    return {
        'true_assignments': true_assignments,
        'cluster_templates': parser.cluster_templates,
    }


if __name__ == '__main__':
    data_config = DataConfigs.Proxifier
    drain_params = (8, 75, 0.4)

    results = run_drain_error_analysis(data_config, drain_params)
    dump_results('drain_error_analysis.p', results)
