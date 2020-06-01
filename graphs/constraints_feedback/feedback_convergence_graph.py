import matplotlib.pyplot as plt
from global_utils import load_results, get_avg

SUBPLOT_DIM = (4, 4)
LEGEND_IDX = 16

results = load_results('feedback_convergence.p')

n_subplots = len(results)

for idx, name in enumerate(results, start=1):
    plt.subplot(*SUBPLOT_DIM, idx)
    plt.title(name)
    plt.grid()
    plt.ylim((0, 1.05))
    plt.tight_layout(0.1)
    imp_rate_results = results[name]['cem']
    for improvement_rate in imp_rate_results:
        acc_vals_samples = imp_rate_results[improvement_rate]['acc']
        avg_acc_vals = get_avg(acc_vals_samples)
        n = len(avg_acc_vals)

        acc_label = 'acc_{}'.format(improvement_rate)
        plt.plot(list(range(n)), avg_acc_vals, label=acc_label)

    if idx == LEGEND_IDX:
        plt.legend()

plt.show()
