import os
import sys
import numpy as np
from tqdm import tqdm
from collections import Counter
from img2vec_pytorch import Img2Vec
from PIL import Image
import bioCaption.models.captionModels.utils as utils


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

    def image_to_vector(self, img2vec, image_id):
        image = Image.open(os.path.join(self.images_dir, image_id))
        image = image.convert('RGB')
        return img2vec.get_vec(image)

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
        train_data = utils.load_data(self.train_dir)
        train_data = utils.get_list_of_words_per_caption(train_data)
        caption_words = train_data['split_captions'].to_list()
        caption_words = [word for caption_list in caption_words for word in caption_list]

        print("The number of total words is:", len(caption_words))
        print("The number of total words is:", len(caption_words))

        # Find the (mean caption length) most frequent words
        frequent_words = Counter(caption_words).most_common(int(round(length)))

        # Join the frequent words to create the frequency caption
        caption = " ".join(f[0] for f in frequent_words)

        print("The caption of most frequent words is:", caption)

        # Load test data and assign the frequency caption to every image to create results
        test_data = utils.load_data(self.test_dir)
        # Dictionary to save the test image ids and the frequency caption
        test_results = {}
        for index, row in test_data.iterrows():
            test_results[row["image_ids"]] = caption
        # Save test results to tsv file
        utils.save_results(test_results, self.results_dir, "most_frequent_word_results.json")
        return test_results

    def one_nn(self, vector_function = utils.average_embedding, cuda=False):
        """
        Nearest Neighbor Baseline: Img2Vec library (https://github.com/christiansafka/img2vec/) is used to obtain
        image embeddings, extracted from ResNet-18. For each test image the cosine similarity with all the training images
        is computed in order to retrieve similar training images.
        The caption of the most similar retrieved image is returned as the generated caption of the test image.

        :param vector_function: function that accepts a numpy array nxm and returns a numpy array 1xm
        :param cuda: Boolean value of whether to use cuda for image embeddings extraction. Default: False
        If a GPU is available pass True
        :return: Dictionary with the results
        """
        img2vec = Img2Vec(cuda=cuda)

        # Load train data
        train_data = utils.load_data(self.train_dir)
        train_images = dict(zip(train_data.image_ids, train_data.caption))

        print("Calculating visual embeddings from train images")
        train_images_vec = {}
        print("Extracting embeddings for all train images...")
        for train_image_lists in tqdm(train_data.img_ids_list):
            image_vectors = []
            for train_image in train_image_lists:
                image_vectors.append(self.image_to_vector(img2vec, train_image))
            train_images_vec[','.join(train_image_lists)] = vector_function(np.array(image_vectors))
        print("Got embeddings for train images.")

        # Load test data
        test_data = utils.load_data(self.test_dir)

        # Save IDs and raw image vectors separately but aligned
        ids = list(train_images_vec.keys())
        raw = np.array([train_images_vec[i] for i in train_images_vec])

        # Normalize image vectors to avoid normalized cosine and use dot
        raw = raw / np.array([np.sum(raw, 1)] * raw.shape[1]).transpose()
        sim_test_results = {}

        for test_image_ids in tqdm(test_data.img_ids_list):
            test_vectors = []
            for test_image in test_image_ids:
                # Get test image embedding
                test_vectors.append(self.image_to_vector(img2vec, test_image))
            vector = vector_function(np.array(test_vectors))
            # Compute cosine similarity with every train image
            vector = vector / np.sum(vector)
            # Clone to do efficient mat mul dot
            test_mat = np.array([vector] * raw.shape[0])
            sims = np.sum(test_mat * raw, 1)
            top1 = np.argmax(sims)
            # Assign the caption of the most similar train image
            sim_test_results[test_image_ids] = train_images[ids[top1]]

        # Save test results to tsv file
        utils.save_results(sim_test_results, self.results_dir, "onenn_results.json")
        return sim_test_results
