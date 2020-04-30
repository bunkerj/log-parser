from global_utils import load_results
from src.helpers.evaluator import Evaluator

results = load_results('drain_error_analysis.p')

evaluator = Evaluator(results['true_assignments'])
evaluator.print_all_discrepancies(results['cluster_templates'])
