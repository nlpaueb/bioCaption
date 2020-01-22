import re
import argparse
import gensim
import pandas as pd
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge


def _bioclean(token):
    return re.sub('[.,?;*!%^&_+():-\[\]{}]', '',
                  token.replace('"', '').replace('/', '').replace('\\', '')
                  .replace("'", '').strip().lower()).split()


class Evaluation:

    def __init__(self, results_dir, gold_dir):
        self.results_dir = results_dir
        self.gold_dir = gold_dir

    def preprocess_captions(self, images_captions):
        """
        :param images_captions: Dictionary with image ids as keys and captions as values
        :return: Dictionary with the processed captions as values
        """
        processed_captions = {}
        # Apply bio clean to data
        for image in images_captions:
            processed_captions[image] = _bioclean(images_captions[image])

        return processed_captions

    def compute_wmd(self, gts, res, bio_path):
        """
        :param gts: Dictionary with the image ids and their gold captions
        :param res: Dictionary with the image ids ant their generated captions
        :param bio_path: Path to the pre-trained biomedical word embeddings
        :print: WMD and WMS scores
        """

        # Preprocess captions
        gts = preprocess_captions(gts)
        res = preprocess_captions(res)

        # Load word embeddings
        bio = gensim.models.KeyedVectors.load_word2vec_format(bio_path, binary=True)
        print("Loaded word embeddings")

        # Calculate WMD for each gts-res captions pair
        print("Calculating wmd for each pair...")
        total_distance = 0
        img_wmds, similarities = {}, {}

        assert len(gts) == len(res)

        for image in gts:
            distance = bio.wmdistance(gts[image].split(), res[image].split())
            similarities[image] = (1. / (1. + distance))
            total_distance = total_distance + distance
            img_wmds[image] = distance

        # calculate mean wmd
        wmd = total_distance / float(len(gts))
        wms = sum(similarities.values()) / float(len(similarities))

        print("WMD =", wmd, ", WMS =", wms)


