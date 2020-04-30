from global_utils import load_results

results = load_results('multinomial_timing.p')

print('Scipy implementation: {}'.format(results['custom_time']))
print('Custom implementation: {}'.format(results['scipy_time']))
