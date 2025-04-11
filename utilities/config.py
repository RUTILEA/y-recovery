import sys
from loguru import logger
import yaml

logger.remove(0)
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss zz}</green> <cyan>{function}</cyan> <level>{message}</level>",
    level="INFO",
)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
