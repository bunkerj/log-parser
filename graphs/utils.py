import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from constants import RESULTS_DIR
from graphs.helpers import autolabel


def show_dataset_comparison_graph(dataset_name, benchmark_accuracies, final_best_accuracies):
    relevant_benchmark_accuracies = {}
    for name in benchmark_accuracies:
        if name in final_best_accuracies:
            relevant_benchmark_accuracies[name] = benchmark_accuracies[name]

    labels = relevant_benchmark_accuracies.keys()
    baseline_accuracies = relevant_benchmark_accuracies.values()
    candidate_accuracies = final_best_accuracies.values()

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, baseline_accuracies, width, label='Baseline')
    rects2 = ax.bar(x + width / 2, candidate_accuracies, width, label='Candidate')

    ax.set_ylabel('Accuracy')
    ax.set_title('{} Accuracies'.format(dataset_name))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(ax, rects1)
    autolabel(ax, rects2)

    fig.tight_layout()

    plt.show()


def load_results(name):
    path = os.path.join(RESULTS_DIR, name)
    return pickle.load(open(path, 'rb'))
