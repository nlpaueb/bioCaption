import os
import sys
import re
import pandas as pd
from collections import Counter
from img2vec.img_to_vec import Img2Vec
from PIL import Image
from tqdm import tqdm
import numpy as np

sys.path.append("..")  # Adds higher directory to python modules path.


class Baselines:
    # Clean for BioASQ
    bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '',
                                t.replace('"', '').replace('/', '').replace('\\', '')
                                .replace("'", '').strip().lower()).split()

    def __init__(self, train_dir, test_dir, results_dir, images_dir=None):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.results_dir = results_dir
        self.images_dir = images_dir

    def most_frequent_word_in_captions(self, length):
        """
        Frequency baseline: uses the frequency of words in the training captions to always generate the same caption.
        The most frequent word always becomes the first word of the caption, the next most frequent word always
        becomes the second word of the caption, etc.
        The number of words in the generated caption is the average length of training captions.

        :param train_path: The path to the train data tsv file with the form: "image \t caption"
        :param test_path: The path to the test data tsv file with the form: "image \t caption"
        :param results_path: The folder in which to save the results file
        :param length: The mean caption length of the train of the train captions
        :return: Dictionary with the results
        """

        # load train data to find most frequent words
        words = []
        with open(self.train_dir, "r") as file:
            for line in file:
                line = line.replace("\n", "").split("\t")
                tokens = Baselines.bioclean(line[1])
                for token in tokens:
                    words.append(token)

        print("The number of total words is:", len(words))

        # Find the (mean caption length) most frequent words
        frequent_words = Counter(words).most_common(int(round(length)))

        # Join the frequent words to create the frequency caption
        caption = " ".join(f[0] for f in frequent_words)

        print("The caption of most frequent words is:", caption)

        # Load test data and assign the frequency caption to every image to create results
        test_data = pd.read_csv(self.test_path, sep="\t", names=["image_ids", "captions"], header=None)
        # Dictionary to save the test image ids and the frequency caption
        test_results = {}
        for index, row in test_data.iterrows():
            test_results[row["image_ids"]] = caption

        # Save test results to tsv file
        df = pd.DataFrame.from_dict(test_results, orient="index")
        df.to_csv(os.path.join(self.results_path, "most_frequent_word_results.tsv"), sep="\t", header=False)

        return test_results

    def onenn(self, cuda=False):
        pass
