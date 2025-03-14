import sys
sys.path.append('/home/ori/pytracking-master/ltr/external/AR')
from ltr.external.AR.ltr.admin.environment import env_settings


class Settings:
    """ Training settings, e.g. the paths to datasets and networks."""
    def __init__(self):
        self.set_default()

    def set_default(self):
        self.env = env_settings()
        self.use_gpu = True


