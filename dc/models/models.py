import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from img2vec_pytorch import Img2Vec
from PIL import Image
import dc.data.data_functions as functions


sys.path.append("..")  # Adds higher directory to python modules path.


class Models:

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


