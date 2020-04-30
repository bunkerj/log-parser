import matplotlib.pyplot as plt
from global_utils import load_results

results = load_results('enhanced_drain_exp.p')

accuracy = results['accuracy']
mins_to_parse = results['mins_to_parse']
mins_to_eval = results['mins_to_eval']
parser = results['parser']

print('Final Drain Accuracy: {}'.format(accuracy))
print('Time to parse: {:.5f} minutes'.format(mins_to_parse))
print('Time to evaluate: {:.5f} minutes'.format(mins_to_eval))
print('Total time taken: {:.5f} minutes'.format(mins_to_parse + mins_to_eval))

parser.plot_dendrogram(leaf_font_size=5,
                       leaf_rotation=67,
                       truncate_mode=None,
                       orientation='right')
plt.show()
