import matplotlib.pyplot as plt
from global_utils import load_results

results = load_results('online_em_results.p')

init_sizes = results['init_sizes']
results_scores = results['scores']
results_timings = results['timings']

plt.subplot(1, 2, 1)
plt.plot(init_sizes, results_timings['cem'])
plt.plot(init_sizes, results_timings['online_cem'])
plt.plot(init_sizes, results_timings['online_em'])
plt.legend(['CEM', 'Online CEM', 'Online EM'])
plt.ylabel('Timing (Seconds)')
plt.xlabel('Sample Size')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(init_sizes, results_scores['cem'])
plt.plot(init_sizes, results_scores['online_cem'])
plt.plot(init_sizes, results_scores['online_em'])
plt.legend(['CEM', 'Online CEM', 'Online EM'])
plt.ylabel('Impurity')
plt.xlabel('Sample Size')
plt.grid()

plt.show()
