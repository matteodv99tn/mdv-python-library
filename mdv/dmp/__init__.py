import logging
import sys

from .. import default_logger_formatter


logger = logging.getLogger("Dmp")
logger.setLevel(logging.DEBUG)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(default_logger_formatter)
logger.addHandler(stdout_handler)

from .demonstration import Demonstration
from .dmp import Dmp
