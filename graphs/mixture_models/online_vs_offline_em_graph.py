import matplotlib.pyplot as plt
from global_utils import load_results

results = load_results('offline_vs_online_em_results.p')

offline_log_likelihood = results['offline_log_likelihood']
online_log_likelihood = results['online_log_likelihood']
epochs = list(range(len(offline_log_likelihood)))

plt.plot(epochs, offline_log_likelihood)
plt.plot(epochs, online_log_likelihood)
plt.legend(['Offline EM', 'Online EM'])
plt.ylabel('Log-likelihood')
plt.xlabel('Epoch')
plt.grid()
plt.show()
