"""
Plots a histogram of impurities to compare the unlabeled (baseline) impurities
against labeled impurity.
"""

from graphs.utils import plot_dataset_comparison_graph
from global_utils import load_results

results = load_results('online_benchmark.p')

impurities_lab = {name: results[name]['unlab'] for name in results}
impurities_unlab = {name: results[name]['lab'] for name in results}

plot_dataset_comparison_graph('Impurity Reduction from Labeling',
                              impurities_lab,
                              impurities_unlab)
    