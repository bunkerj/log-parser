from src.methods.iplom import Iplom
from src.constants import BGL_FILE_PATH

iplom = Iplom(BGL_FILE_PATH, 0.3, 0.8)
result = iplom.parse()
# templates = iplom.get_templates()
