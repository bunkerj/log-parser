import numpy as np
import matplotlib.pyplot as plt


def plot_dataset_comparison_graph(title, benchmark_accuracies,
                                  final_best_accuracies, annotate=True):
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

    if annotate:
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
                    ha='center', va='bottom', rotation=90)


def plot_morris_method_graph(sensitivity_indices, title):
    plt.scatter(sensitivity_indices['mu_star'], sensitivity_indices['sigma'],
                marker='x')

    for idx, name in enumerate(sensitivity_indices['names']):
        point = (sensitivity_indices['mu_star'][idx],
                 sensitivity_indices['sigma'][idx])
        plt.annotate(name, point, fontsize=10, ha='center', va='bottom')

    plt.ylabel(r'$\sigma$')
    plt.xlabel(r'$\mu^*$')
    plt.title(title)
    plt.grid()


def plot_mean_with_ci(t, mu1, mu2, s1, s2):
    plt.plot(t, mu1, lw=2, label='Avg Unlab Impurity', color='blue')
    plt.plot(t, mu2, lw=2, label='Avg Lab Impurity', color='green')
    plt.fill_between(t, mu1 + 2 * s1, mu1 - 2 * s1, facecolor='blue',
                     alpha=0.5)
    plt.fill_between(t, mu2 + 2 * s2, mu2 - 2 * s2, facecolor='green',
                     alpha=0.5)
    plt.grid()
