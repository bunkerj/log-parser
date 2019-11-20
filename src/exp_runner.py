from src.methods.iplom import Iplom
from src.constants import BGL_FILE_PATH

iplom = Iplom(BGL_FILE_PATH,
              file_threshold=0.1,
              partition_threshold=0,
              lower_bound=0.1,
              upper_bound=0.9,
              goodness_threshold=0.34)

result = iplom.parse()
