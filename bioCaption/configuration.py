import logging
from bioCaption import default_config as config


def get_logger():
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S', level=config.LOG_LEVEL)
    return logger
