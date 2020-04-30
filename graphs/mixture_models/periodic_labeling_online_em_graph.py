import matplotlib.pyplot as plt
from global_utils import load_results

result_filename = 'periodic_labeling_online_em.p'
results = load_results(result_filename)

iterations = list(results['lab_online_em'].keys())
lab_em_impurities = list(results['lab_online_em'].values())
unlab_em_impurities = list(results['unlab_online_em'].values())
lab_cem_impurities = list(results['lab_online_cem'].values())

plt.plot(iterations, lab_em_impurities)
plt.plot(iterations, lab_cem_impurities)
plt.plot(iterations, unlab_em_impurities)
plt.legend(['Online Labeled EM', 'Online Labeled CEM', 'Online Unlabeled EM'])
plt.xlabel('Iteration')
plt.ylabel('Impurity')
plt.grid()
plt.show()
