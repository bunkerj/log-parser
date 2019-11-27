import numpy as np
import matplotlib.pyplot as plt
from graphs.helpers import autolabel


def print_dataset_comparison(benchmark_baseline_accuracies, final_best_accuracies):
    labels = benchmark_baseline_accuracies.keys()
    baseline_accuracies = benchmark_baseline_accuracies.values()
    candidate_accuracies = final_best_accuracies.values()

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, baseline_accuracies, width, label='Baseline')
    rects2 = ax.bar(x + width / 2, candidate_accuracies, width, label='Candidate')

    ax.set_ylabel('Accuracy')
    ax.set_title('IPLoM Accuracies')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(ax, rects1)
    autolabel(ax, rects2)

    fig.tight_layout()

    plt.show()
