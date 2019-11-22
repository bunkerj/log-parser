from src.methods.iplom import Iplom
from src.constants import BGL_FILE_PATH, BGL_STRUCTURED_PATH
from src.helpers.evaluator import Evaluator

iplom = Iplom(BGL_STRUCTURED_PATH,
              file_threshold=0.05,
              partition_threshold=0.05,
              lower_bound=0.1,
              upper_bound=0.9,
              goodness_threshold=0.34)

iplom.parse()
iplom.print_cluster_templates()

evaluator = Evaluator(BGL_STRUCTURED_PATH, iplom.cluster_templates)
iplom_bgl_accuracy = evaluator.evaluate()

print('IPLoM BGL Accuracy: {}'.format(iplom_bgl_accuracy))
