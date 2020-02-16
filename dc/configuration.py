import logging
import os

DOWNLOAD_PATH = os.getcwd()
log_level = logging.DEBUG


def get_logger():
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S', level=log_level)
    return logger
