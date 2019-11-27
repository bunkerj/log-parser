import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from constants import RESULTS_DIR


def show_dataset_comparison_graph(title, benchmark_accuracies, final_best_accuracies):
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
    rects1 = ax.bar(x - width / 2, baseline_accuracies, width,
                    label='Baseline', color=np.random.rand(3, ))
    rects2 = ax.bar(x + width / 2, candidate_accuracies, width,
                    label='Candidate', color=np.random.rand(3, ))

    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(ax, rects1)
    autolabel(ax, rects2)

    fig.tight_layout()

    plt.show()


def autolabel(ax, rects):
    """
    Attach a text label above each bar to display its height.
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def load_results(name):
    path = os.path.join(RESULTS_DIR, name)
    return pickle.load(open(path, 'rb'))
