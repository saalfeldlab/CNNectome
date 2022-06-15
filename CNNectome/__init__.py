from pkg_resources import resource_filename

with open(resource_filename("CNNectome", "VERSION"), "r") as version_file:
    __version__ = version_file.read().strip()
del resource_filename
from . import networks
from . import postprocessing
from . import training
from . import utils
from . import validation
from . import visualization
