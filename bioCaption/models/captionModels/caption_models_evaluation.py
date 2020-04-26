import os
import gensim
import re
import pandas as pd
from bioCaption.data.downloads import download_bio_embeddings
import json
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from bioCaption.configuration import get_logger


def _bioclean(caption):
    return re.sub('[.,?;*!%^&_+():-\[\]{}]',
                  '',
                  caption.replace('"', '')
                  .replace('/', '')
                  .replace('\\', '')
                  .replace("'", '')
                  .strip()
                  .lower())


class CaptionsEvaluation:

    logger = get_logger()

    def __init__(self, gold_dir, results_dir):
        self.results_dir = results_dir
        self.gold_dir = gold_dir
        self.gold_data = {}
        self.result_data = {}
        self.result_data = {}

    def _load_data(self):
        gold_csv = pd.read_csv(self.gold_dir, sep="\t", header=None, names=["image_ids", "captions"],
                               encoding='utf-8', engine='python')
        self.gold_data = dict(zip(gold_csv.image_ids, gold_csv.captions))
        with open(self.results_dir) as json_file:
            self.result_data = json.load(json_file)

    def _preprocess_captions(self, images_caption_dict):
        """
        :param images_caption_dict: Dictionary with image ids as keys and captions as values
        :return: Dictionary with the processed captions as values and the id of the images as
        key
        """
        processed_captions_dict = {}
        for image_id in images_caption_dict:
            processed_captions_dict[image_id] = [_bioclean(images_caption_dict[image_id])]
        return processed_captions_dict

    def compute_WMD(self, bio_path, embeddings_file='pubmed2018_w2v_200D.bin'):
        """Word Mover's Distance computes the minimum cumulative cost required to move all word embeddings of one caption
        to aligned word embeddings of the other caption. We used Gensim's implementation of WMD (https://goo.gl/epzecP)
        and biomedical word2vec embeddings (https://archive.org/details/pubmed2018_w2v_200D.tar).
        WMD scores are also expressed as similarity values: WMS = (1 + WMD)^-1

        :param embeddings_file: the binary file with the embeddings
        :param gts: Dictionary with the image ids and their gold captions
        :param res: Dictionary with the image ids ant their generated captions
        :param bio_path: Path to the pre-trained biomedical word embeddings,
                        if the ebmeddings are not there, they will be downloaded.
        :print: WMD and WMS scores
        """
        # load the csv files, containing the results and gold data.
        self.logger.info("Loading data")
        self._load_data()

        # Preprocess captions
        self.logger.info("Preprocessing captions")
        self.gold_data = self._preprocess_captions(self.gold_data)
        self.result_data = self._preprocess_captions(self.result_data)

        # Load word embeddings
        self.logger.info("Trying to load word embeddings....")
        if not os.path.exists(os.path.join(bio_path, embeddings_file)):
            self.logger.info(f"Bio embeddings do not exists. Will try to download them "
                             "in {os.path.join(config.DOWNLOAD_PATH, bio_path)}")
            download_bio_embeddings(bio_path)

        bio = gensim.models.KeyedVectors.\
            load_word2vec_format(os.path.join(bio_path, embeddings_file), binary=True)

        # Calculate WMD for each gts-res captions pair
        self.logger.info("Calculating WMD for each pair...")
        total_distance = 0
        img_wmds, similarities = {}, {}

        if len(self.gold_data) == len(self.result_data):
            for image in self.gold_data:
                self.logger.debug(self.gold_data[image])
                distance = bio.wmdistance(self.gold_data[image][0], self.result_data[image][0])
                similarities[image] = (1. / (1. + distance))
                total_distance = total_distance + distance
                img_wmds[image] = distance

            # calculate mean wmd
            wmd = total_distance / float(len(self.gold_data))
            wms = sum(similarities.values()) / float(len(similarities))

            print("WMD =", wmd, ", WMS =", wms)
        else:
            self.logger.error("Gold data len={0} and results data len={1} do not equal size"
                              .format(len(self.gold_data), len(self.result_data)))

    def compute_ms_coco(self):
        """Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)
        :param gts: Dictionary with the image ids and their gold captions,
        :param res: Dictionary with the image ids and their generated captions
        :print: Evaluation score (the mean of the scores of all the instances) for each measure
        """

        # load the csv files, containing the results and gold data.
        self.logger.info("Loading data")
        self._load_data()

        # Preprocess captions
        self.logger.info("Preprocessing captions")
        self.gold_data = self._preprocess_captions(self.gold_data)
        self.result_data = self._preprocess_captions(self.result_data)
        if len(self.gold_data) == len(self.result_data):
            # Set up scorers
            scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Meteor(), "METEOR"),
                (Rouge(), "ROUGE_L")
            ]

            # Compute score for each metric
            self.logger.info("Computing COCO score.")
            for scorer, method in scorers:
                print("Computing", scorer.method(), "...")
                score, scores = scorer.compute_score(self.gold_data, self.result_data)
                if type(method) == list:
                    for sc, m in zip(score, method):
                        print("%s : %0.3f" % (m, sc))
                else:
                    print("%s : %0.3f" % (method, score))
        else:
            self.logger.error("Gold data len={0} and results data len={1} have not equal size"
                              .format(len(self.gold_data), len(self.result_data)))
