import matplotlib.pyplot as plt
from global_utils import load_results

SUBPLOT_DIM = (4, 4)
LEGEND_IDX = 13

results = load_results('feedback_convergence.p')

n_subplots = len(results)

for idx, name in enumerate(results, start=1):
    plt.subplot(*SUBPLOT_DIM, idx)
    plt.title(name)
    plt.grid()
    plt.tight_layout()
    for improvement_rate in results[name]:
        acc_vals = results[name][improvement_rate]['acc']
        n = len(acc_vals)

        acc_label = 'acc_{}'.format(improvement_rate)
        plt.plot(list(range(n)), acc_vals, label=acc_label)

    if idx == LEGEND_IDX:
        plt.legend()

plt.show()
