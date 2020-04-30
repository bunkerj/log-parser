"""
Plot the Drain accuracy for a specific dataset (DATA_CONFIG) evaluated at every
increment (JUMP_SIZE). This procedure is repeated several times (N_RUNS) and
each run is performed on a shuffled version of the dataset.
"""
import random
from global_utils import dump_results
from src.helpers.data_manager import DataManager
from src.parsers.drain import Drain
from src.data_config import DataConfigs
from src.helpers.evaluator import Evaluator


def run_drain_accuracy_over_num_logs(jump_size, n_runs, data_config, name):
    data_manager = DataManager(data_config)
    tokenized_log_entries = data_manager.get_tokenized_log_entries()
    true_assignments = data_manager.get_true_assignments()

    accuracies = []
    end_indices = []
    for run in range(n_runs):
        accuracy_idx = 0
        rand_float = random.random()
        random.shuffle(tokenized_log_entries, lambda: rand_float)
        random.shuffle(true_assignments, lambda: rand_float)
        parser = Drain(tokenized_log_entries, 3, 100, 0.5)
        for end_idx in range(1, len(tokenized_log_entries) + 1):
            parser.single_parse()
            if end_idx % jump_size == 0 or end_idx == len(
                    tokenized_log_entries):
                parser.discover_cluster_templates()
                evaluator = Evaluator(true_assignments[:end_idx])
                accuracy = evaluator.evaluate(parser.cluster_templates)
                if run == 0:
                    accuracies.append(accuracy / n_runs)
                    end_indices.append(end_idx)
                else:
                    accuracies[accuracy_idx] += accuracy / n_runs
                accuracy_idx += 1

    print('Final Drain Accuracy: {}'.format(accuracies[-1]))

    results = {
        'accuracies': accuracies,
        'end_indices': end_indices,
    }

    dump_results(name, results)


if __name__ == '__main__':
    jump_size = 10
    n_runs = 20
    data_config = DataConfigs.BGL
    name = 'drain_accuracy_over_num_logs.p'

    run_drain_accuracy_over_num_logs(jump_size, n_runs, data_config, name)
