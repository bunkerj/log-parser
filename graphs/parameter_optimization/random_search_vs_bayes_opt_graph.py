import matplotlib.pyplot as plt
from global_utils import load_results

random_search_history = load_results('drain_bayesian_optimization.p')
bayes_opt_history = load_results('drain_single_random_search.p')
accuracy_indices = range(1, len(random_search_history) + 1)

plt.plot(accuracy_indices, random_search_history)
plt.plot(accuracy_indices, bayes_opt_history)
plt.legend(['Random Search', 'Bayesian Optimization'])
plt.title('Averaged Drain Accuracy Over Number of Random Samples')
plt.ylabel('Averaged Accuracy')
plt.xlabel('Number of Samples')
plt.grid()
plt.show()
