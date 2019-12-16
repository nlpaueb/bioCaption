import gensim
import re
import argparse
import pandas as pd

def preprocess_captions(images_captions):
    """

    :param images_captions: Dictionary with image ids as keys and captions as values
    :return: Dictionary with the processed captions as values
    """

    # Clean for BioASQ
    bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '',
                                t.replace('"', '').replace('/', '').replace('\\', '').replace("'",
                                                                                              '').strip().lower())
    pr_captions = {}
    # Apply bio clean to data
    for image in images_captions:
        pr_captions[image] = bioclean(images_captions[image])

    return pr_captions