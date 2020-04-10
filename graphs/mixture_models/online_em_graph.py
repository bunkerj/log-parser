import matplotlib.pyplot as plt
from global_utils import load_results

results = load_results('online_em_results.p')

sample_sizes = results['sample_sizes']
results_acc = results['accuracies']
results_tim = results['timings']

plt.subplot(1, 2, 1)
plt.plot(sample_sizes, results_tim['cem'])
plt.plot(sample_sizes, results_tim['online_cem'])
plt.plot(sample_sizes, results_tim['online_em'])
plt.legend(['CEM', 'Online CEM', 'Online EM'])
plt.ylabel('Timing (Seconds)')
plt.xlabel('Sample Size')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(sample_sizes, results_acc['cem'])
plt.plot(sample_sizes, results_acc['online_cem'])
plt.plot(sample_sizes, results_acc['online_em'])
plt.legend(['CEM', 'Online CEM', 'Online EM'])
plt.ylabel('Accuracy (%)')
plt.xlabel('Sample Size')
plt.grid()

plt.show()
