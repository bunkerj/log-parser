from graphs.utils import plot_dataset_comparison_graph
from global_utils import load_results

results = load_results('feedback_using_constraints.p')
plot_dataset_comparison_graph('Accuracies',
                              results['drain'],
                              results['mmo'])
