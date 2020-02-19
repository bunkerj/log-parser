from graphs.utils import load_results
import matplotlib.pyplot as plt

results = load_results('feedback_evaluation.p')

label_counts = results['label_counts']
labeled_impurities = results['labeled_impurities']
unlabeled_impurities = results['unlabeled_impurities']

plt.plot(label_counts, labeled_impurities)
plt.plot(label_counts, unlabeled_impurities)
plt.title('Impurity Over Number of Labels')
plt.legend(['Labeled Impurity', 'Unlabeled Impurity'])
plt.xlabel('Number of Labels')
plt.ylabel('Impurity')
plt.grid()
plt.show()
