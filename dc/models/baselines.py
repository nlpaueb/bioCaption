import os
import sys
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from img2vec_pytorch import Img2Vec
from PIL import Image

sys.path.append("..")  # Adds higher directory to python modules path.


class Baselines:
    def __init__(self, train_dir, test_dir, images_dir, results_dir):
        """
        :param train_dir: The directory to the train data tsv file with the form: "image \t caption"
        :param test_dir: The directory to the test data tsv file with the form: "image \t caption"
        :param images_dir: : The folder in the dataset that contains the images
         with the form: "[dataset_name]_images".
        :param results_dir: The folder in which to save the results file
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.images_dir = images_dir
        self.results_dir = results_dir

    # Clean for BioASQ
    def bioclean(self, t):
        return re.sub('[.,?;*!%^&_+():-\[\]{}]', '',
                      t.replace('"', '').replace('/', '').replace('\\', '')
                      .replace("'", '').strip().lower()).split()

    def most_frequent_word_in_captions(self, length):
        """
        Frequency baseline: uses the frequency of words in the training captions to always generate the same caption.
        The most frequent word always becomes the first word of the caption, the next most frequent word always
        becomes the second word of the caption, etc.
        The number of words in the generated caption is the average length of training captions.

        :param length: The mean caption length of the train of the train captions
        :return: Dictionary with the results
        """

        # load train data to find most frequent words
        words = []
        with open(self.train_dir, "r") as file:
            for line in file:
                line = line.replace("\n", "").split("\t")
                tokens = self.bioclean(line[1])
                for token in tokens:
                    words.append(token)

        print("The number of total words is:", len(words))

        # Find the (mean caption length) most frequent words
        frequent_words = Counter(words).most_common(int(round(length)))

        # Join the frequent words to create the frequency caption
        caption = " ".join(f[0] for f in frequent_words)

        print("The caption of most frequent words is:", caption)

        # Load test data and assign the frequency caption to every image to create results
        test_data = pd.read_csv(self.test_dir, sep="\t", names=["image_ids", "captions"], header=None)
        # Dictionary to save the test image ids and the frequency caption
        test_results = {}
        for index, row in test_data.iterrows():
            test_results[row["image_ids"]] = caption

        # Save test results to tsv file
        df = pd.DataFrame.from_dict(test_results, orient="index")
        df.to_csv(os.path.join(self.results_dir, "most_frequent_word_results.tsv"), sep="\t", header=False)

        return test_results

    def one_nn(self, cuda=False):
        """
        Nearest Neighbor Baseline: Img2Vec library (https://github.com/christiansafka/img2vec/) is used to obtain
        image embeddings, extracted from ResNet-18. For each test image the cosine similarity with all the training images
        is computed in order to retrieve similar training images.
        The caption of the most similar retrieved image is returned as the generated caption of the test image.

        :param cuda: Boolean value of whether to use cuda for image embeddings extraction. Default: False
        If a GPU is available pass True
        :return: Dictionary with the results
        """

        img2vec = Img2Vec(cuda=cuda)

        # Load train data
        train_data = pd.read_csv(self.train_dir, sep="\t", header=None)
        train_data.columns = ["id", "caption"]
        train_images = dict(zip(train_data.id, train_data.caption))

        # Get embeddings of train images
        print("Calculating visual embeddings from train images")
        train_images_vec = {}
        print("Extracting embeddings for all train images...")
        for train_image in tqdm(train_data.id):
            image = Image.open(os.path.join(self.images_dir, train_image))
            image = image.convert('RGB')
            vec = img2vec.get_vec(image)
            train_images_vec[train_image] = vec
        print("Got embeddings for train images.")

        # Load test data
        test_data = pd.read_csv(self.test_dir, sep="\t", header=None)
        test_data.columns = ["id", "caption"]

        # Save IDs and raw image vectors separately but aligned
        ids = [i for i in train_images_vec]
        raw = np.array([train_images_vec[i] for i in train_images_vec])

        # Normalize image vectors to avoid normalized cosine and use dot
        raw = raw / np.array([np.sum(raw, 1)] * raw.shape[1]).transpose()
        sim_test_results = {}

        for test_image in tqdm(test_data.id):
            # Get test image embedding
            image = Image.open(os.path.join(self.images_dir, test_image))
            image = image.convert('RGB')
            vec = img2vec.get_vec(image)
            # Compute cosine similarity with every train image
            vec = vec / np.sum(vec)
            # Clone to do efficient mat mul dot
            test_mat = np.array([vec] * raw.shape[0])
            sims = np.sum(test_mat * raw, 1)
            top1 = np.argmax(sims)
            # Assign the caption of the most similar train image
            sim_test_results[test_image] = train_images[ids[top1]]

        # Save test results to tsv file
        df = pd.DataFrame.from_dict(sim_test_results, orient="index")
        df.to_csv(os.path.join(self.results_dir, "onenn_results.tsv"), sep="\t", header=False)

        return sim_test_results
