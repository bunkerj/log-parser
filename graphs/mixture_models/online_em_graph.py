import matplotlib.pyplot as plt
from global_utils import load_results

results = load_results('online_em.p')

training_sizes = results['training_sizes']
results_scores = results['scores']
results_timings = results['timings']

plt.subplot(1, 2, 1)
plt.plot(training_sizes, results_timings['em'])
plt.plot(training_sizes, results_timings['cem'])
plt.plot(training_sizes, results_timings['online_cem'])
plt.plot(training_sizes, results_timings['online_em'])
plt.legend(['EM', 'CEM', 'Online CEM', 'Online EM'])
plt.ylabel('Average Timing (Seconds)')
plt.xlabel('Training Size')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(training_sizes, results_scores['em'])
plt.plot(training_sizes, results_scores['cem'])
plt.plot(training_sizes, results_scores['online_cem'])
plt.plot(training_sizes, results_scores['online_em'])
plt.ylabel('Average Impurity')
plt.xlabel('Training Size')
plt.grid()

plt.show()
