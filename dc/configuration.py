import os
import sys
import json
import configparser
import logging
import datetime
from configparser import ParsingError
from os.path import join as path_join, abspath

ROOT_PATH = abspath(os.getcwd())
#LOG_DIR = abspath(path_join(os.getcwd(), 'log'))
LOG_DIR='/log'


def read_configuration():
    # path relative to run.py
    if os.path.exists(abspath(path_join(ROOT_PATH, 'ipa-rest-resource-management.ini'))):
        conf_dir = abspath(path_join(ROOT_PATH, 'ipa-rest-resource-management.ini'))
    # else path needs to be relative to the tests
    else:
        conf_dir = abspath(path_join(ROOT_PATH, os.pardir, os.pardir, 'ipa-rest-resource-management.ini'))
    config = configparser.ConfigParser()
    config.read(conf_dir)
    return config


def get_logger(log_level=logging.DEBUG):
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S', level=log_level)
    return logger


