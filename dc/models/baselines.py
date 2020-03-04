import os
import sys
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from img2vec_pytorch import Img2Vec
from PIL import Image


sys.path.append("..")  # Adds higher directory to python modules path.


class Baselines:

    def __init__(self, train_dir, test_dir, images_dir, results_dir):
        """
        :param train_dir: The directory to the train data tsv file with the form: "[image1,image2] \t caption"
        :param test_dir: The directory to the test data tsv file with the form: "[image1, imager2] \t caption"
        :param images_dir: : The folder in the dataset that contains the images
         with the form: "[dataset_name]_images".
        :param results_dir: The folder in which to save the results file
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.images_dir = images_dir
        self.results_dir = results_dir
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    # Clean for BioASQ
    def _bioclean(self, token):
        return re.sub('[.,?;*!%^&_+():-\[\]{}]', '',
                      token.replace('"', '').replace('/', '').replace('\\', '')
                      .replace("'", '').strip().lower()).split()

    def most_frequent_word_in_captions(self, length=5):
        """
        Frequency baseline: uses the frequency of words in the training captions to always generate the same caption.
        The most frequent word always becomes the first word of the caption, the next most frequent word always
        becomes the second word of the caption, etc.
        The number of words in the generated caption is the average length of training captions.

        :param length: The mean caption length of the train of the train captions
        :return: Dictionary with the results
        """

        # load train data to find most frequent words
        caption_words = []
        with open(self.train_dir, "r") as file:
            for line in file:
                image_ids, caption = line.replace("\n", "").split("\t")
                #list of image ids
                images_ids = images_ids.split(',')
                caption_tokens = self._bioclean(caption)
                for token in caption_tokens:
                    caption_words.append(token)

        print("The number of total words is:", len(caption_words))
        print("The number of total words is:", len(caption_words))

        # Find the (mean caption length) most frequent words
        frequent_words = Counter(caption_words).most_common(int(round(length)))

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

    def one_nn(self, embeddings_func=util_functions.average, cuda=False):
        """
        Nearest Neighbor Baseline: Img2Vec library (https://github.com/christiansafka/img2vec/) is used to obtain
        image embeddings, extracted from ResNet-18. For each test image the cosine similarity with all the training images
        is computed in order to retrieve similar training images.
        The caption of the most similar retrieved image is returned as the generated caption of the test image.

        :param embeddings_func: function that accepts a numpy array nxm and returns a numpy array 1xm
        :param cuda: Boolean value of whether to use cuda for image embeddings extraction. Default: False
        If a GPU is available pass True
        :return: Dictionary with the results
        """

        img2vec = Img2Vec(cuda=cuda)

        # Load train data
        train_data = pd.read_csv(self.train_dir, sep="\t", header=None)
        train_data.columns = ["ids", "caption"]
        train_images = dict(zip(train_data.ids, train_data.caption))

        # Get embeddings of train images
        print("Calculating visual embeddings from train images")
        train_images_vec = {}
        print("Extracting embeddings for all train images...")
        for train_image_ids in tqdm(train_data.id):
            vector = self.images_to_vector(train_image_ids.split(','), img2vec, embeddings_func)
            train_images_vec[train_image_ids] = vector
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

        for test_image_ids in tqdm(test_data.id):
            # Get test image embedding
            vector = self.images_to_vector(test_image_ids.split(','), img2vec, embeddings_func)
            # Compute cosine similarity with every train image
            vector = vector / np.sum(vector)
            # Clone to do efficient mat mul dot
            test_mat = np.array([vector] * raw.shape[0])
            sims = np.sum(test_mat * raw, 1)
            top1 = np.argmax(sims)
            # Assign the caption of the most similar train image
            sim_test_results[test_image_ids] = train_images[ids[top1]]

        # Save test results to tsv file
        df = pd.DataFrame.from_dict(sim_test_results, orient="index")
        df.to_csv(os.path.join(self.results_dir, "onenn_results.tsv"), sep="\t", header=False)

        return sim_test_results

    def average(self, embeddings_matrix):
        return np.average(embeddings_matrix, axis=0)

    def images_to_vector(self, image_ids_list, img2vec, embeddings_func):
        vectors = []
        for image_id in image_ids_list:
            image = Image.open(os.path.join(self.images_dir, image_id))
            image = image.convert('RGB')
            vectors.append(img2vec.get_vec(image))
        return embeddings_func(vectors)
