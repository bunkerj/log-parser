from src.helpers.template_assigner import TemplateAssigner
from src.data_config import DataConfigs

template_assigner = TemplateAssigner(DataConfigs.BGL_FULL)
template_assigner.write_assignments()
