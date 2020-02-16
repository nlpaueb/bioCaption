import re
import gensim
import pandas as pd
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from dc.configuration import logger



def _bioclean(token):
    return re.sub('[.,?;*!%^&_+():-\[\]{}]', '',
                  token.replace('"', '').replace('/', '').replace('\\', '')
                  .replace("'", '').strip().lower())


class Evaluation:
    
    logger = logger()
    
    def __init__(self, results_dir, gold_dir):
        self.results_dir = results_dir
        self.gold_dir = gold_dir
        self.gold_data = {}
        self.result_data = {}

    def _load_data(self):
        gold_csv = pd.read_csv(self.gold_dir, sep="\t", header=None, names=["image_ids", "captions"],
                               encoding='utf-8', engine='python')
        self.gold_data = dict(zip(gold_csv.image_ids, gold_csv.captions))

        results_csv = pd.read_csv(self.results_dir, sep="\t", header=None, names=["image_ids", "captions"],
                                  encoding='utf-8', engine='python')
        self.result_data = dict(zip(results_csv.image_ids, results_csv.captions))

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

    def compute_WMD(self, bio_path):
        """Word Mover's Distance computes the minimum cumulative cost required to move all word embeddings of one caption
        to aligned word embeddings of the other caption. We used Gensim's implementation of WMD (https://goo.gl/epzecP)
        and biomedical word2vec embeddings (https://archive.org/details/pubmed2018_w2v_200D.tar).
        WMD scores are also expressed as similarity values: WMS = (1 + WMD)^-1

        :param gts: Dictionary with the image ids and their gold captions
        :param res: Dictionary with the image ids ant their generated captions
        :param bio_path: Path to the pre-trained biomedical word embeddings
        :print: WMD and WMS scores
        """
        # load the csv files, containing the results and gold data.
        self._logger.info("Loading data")
        self.load_data()

        # Preprocess captions
        self._logger.info("Preprocessing captions")
        self.gold_data = self.preprocess_captions(self.gold_data)
        self.result_data = self.preprocess_captions(self.result_data)

        # Load word embeddings
        self._logger.info("Loading word embeddings....")
        bio = gensim.models.KeyedVectors.load_word2vec_format(bio_path, binary=True)
        self._logger.info("Loaded!")

        # Calculate WMD for each gts-res captions pair
        self._logger.info("Calculating WMD for each pair...")
        total_distance = 0
        img_wmds, similarities = {}, {}

        if len(self.gold_data) == len(self.result_data):
            for image in self.gold_data:
                print(self.gold_data[image])
                distance = bio.wmdistance(self.gold_data[image][0].split(), self.result_data[image][0].split())
                similarities[image] = (1. / (1. + distance))
                total_distance = total_distance + distance
                img_wmds[image] = distance

            # calculate mean wmd
            wmd = total_distance / float(len(self.gold_data ))
            wms = sum(similarities.values()) / float(len(similarities))

            print("WMD =", wmd, ", WMS =", wms)
        else:
            self._logger.error("Gold data len={0} and results data len={1} have not equal size"
                         .format(len(self.gold_data), len(self.result_data)))

    def compute_ms_coco(self):
        """Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)
        :param gts: Dictionary with the image ids and their gold captions,
        :param res: Dictionary with the image ids ant their generated captions
        :print: Evaluation score (the mean of the scores of all the instances) for each measure
        """

        # load the csv files, containing the results and gold data.
        self._logger.info("Loading data")
        self.load_data()

        # Preprocess captions
        self._logger.info("Preprocessing captions")
        self.gold_data = self.preprocess_captions(self.gold_data)
        self.result_data = self.preprocess_captions(self.result_data)
        if len(self.gold_data) == len(self.result_data):
            # Set up scorers
            scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Meteor(), "METEOR"),
                (Rouge(), "ROUGE_L")
            ]

            # Compute score for each metric
            self._logger.info("Computing COCO score.")
            for scorer, method in scorers:
                print("Computing", scorer.method(), "...")
                score, scores = scorer.compute_score(self.gold_data, self.result_data)
                if type(method) == list:
                    for sc, m in zip(score, method):
                        print("%s : %0.3f" % (m, sc))
                else:
                    print("%s : %0.3f" % (method, score))
        else:
            self._logger.error("Gold data len={0} and results data len={1} have not equal size"
                         .format(len(self.gold_data), len(self.result_data)))


