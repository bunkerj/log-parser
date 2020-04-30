import matplotlib.pyplot as plt
from global_utils import load_results

DIM = (2, 4)
results = load_results('offline_vs_online_em.p')

for idx, name in enumerate(results):
    offline_log_likelihood = results[name]['offline']
    online_log_likelihood = results[name]['online']
    epochs = list(range(len(offline_log_likelihood)))

    plt.subplot(*DIM, idx + 1)
    plt.plot(epochs, offline_log_likelihood)
    plt.plot(epochs, online_log_likelihood)
    plt.title(name)
    plt.grid()

    if idx == ((DIM[0] - 1) * DIM[1]):
        plt.ylabel('Log-likelihood')
        plt.xlabel('Epoch')
        plt.legend(['Offline EM', 'Online EM'])

plt.show()
