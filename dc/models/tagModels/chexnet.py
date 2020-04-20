# CheXNet extended for extreme biomedical multi label classification
from numpy.random import seed
seed(42)
#from tensorflow import
#set_random_seed(42)
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import preprocess_input
from keras.models import Model
from keras.layers import Dense
from keras import backend as K
from keras.preprocessing import image
from keras.initializers import glorot_uniform
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import os
import json
import math
from tqdm import tqdm
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger, ReduceLROnPlateau
from math import ceil
import random
import sys
from keras.models import load_model

from sklearn.metrics import f1_score

random.seed(42)

import os
import sys
import numpy as np
from tqdm import tqdm
from collections import Counter
from keras.applications.densenet import DenseNet121
from keras.models import Model
import keras.applications.densenet as densenet
from keras.preprocessing import image
from sklearn.metrics import f1_score

sys.path.append("..")  # Adds higher directory to python modules path.


class Chexnet:

    def __init__(self, data_dir, images_dir, results_dir, split_ratio=[0.5, 0.4, 0.1]):
        """
        :param train_dir: The directory to the train data tsv file with the form: "[image1,image2] \t caption"
        :param test_dir: The directory to the test data tsv file with the form: "[image1, imager2] \t caption"
        :param images_dir: : The folder in the dataset that contains the images
         with the form: "[dataset_name]_images".
        :param results_dir: The folder in which to save the results file
        """
        self.images_dir = images_dir
        self.results_dir = results_dir
        self.train_data = {}
        self.test_data = {}
        self.val_data = {}
        self.train_concepts = []
        self.val_concepts = []
        self.data_dir = data_dir
        self.ids = []
        self.raw = []
        self.split_ratio = split_ratio
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def _create_dataset(self, path):
        with open(path) as json_file:
            data = json.load(json_file)
            keys = list(data.keys())
            train_pointer = math.ceil(self.split_ratio[0] * len(keys))
            test_pointer = math.ceil(self.split_ratio[1] * len(keys))
            val_pointer = math.ceil(self.split_ratio[2] * len(keys))
            train_keys = keys[:train_pointer]
            test_keys = keys[train_pointer:train_pointer + test_pointer]
            val_keys = keys[train_pointer + test_pointer:val_pointer + train_pointer + test_pointer]
            train, test, val = {}, {}, {}
            train_concepts, val_concepts = [], []
            for key in train_keys:
                train[key] = data[key]
                train_concepts.extend(data[key])
            for key in test_keys:
                test[key] = data[key]
            for key in val_keys:
                val[key] = data[key]
                val_concepts.extend(data[key])
            self.train_data = train
            self.val_data = val
            self.test_data = test
            self.train_concepts = list(set(train_concepts))
            self.val_concepts = list(set(val_concepts))

    def load_data(self, data, concepts_list):
        x_data, y_data = [], []
        # read the data file
        for img_id in tqdm(data.keys()):
            image_path = os.path.join(self.images_dir, img_id)
            img = image.load_img(image_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = preprocess_input(x)
            # encode the tags
            concepts = np.zeros(len(concepts_list), dtype=int)
            if len(data[img_id]) != 0:
                image_concepts = data[img_id]
            else:
                image_concepts = []
            for i in range(0, len(concepts_list)):
                # if the tag is assigned to the image put 1 in its position in the true binary vector
                if concepts_list[i] in image_concepts:
                    concepts[i] = 1
            x_data.append(x)
            y_data.append(concepts)
        #creates images and labels
        return np.array(x_data), np.array(y_data)

    def load_test_images(self, path_to_images, filepath):
        x_test = []
        for img_id, tags in tqdm(self.test_data):
            image_path = os.path.join(self.images_dir, img_id)
            img = image.load_img(image_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = preprocess_input(x)
            x_test.append(x)
        return np.array(x_test)

    # calculates f1 score between y_true and y_pred
    def evaluate_f1(self, gt_pairs, candidate_pairs):
        # Concept stats
        min_concepts = sys.maxsize
        max_concepts = 0
        total_concepts = 0
        concepts_distrib = {}
        # Define max score and current score
        max_score = len(gt_pairs)
        current_score = 0
        # Check there are the same number of pairs between candidate and ground truth
        if len(candidate_pairs) != len(gt_pairs):
            print('ERROR : Candidate does not contain the same number of entries as the ground truth!')
            exit(1)
        for image_key in candidate_pairs:
            # Get candidate and GT concepts
            candidate_concepts = candidate_pairs[image_key].upper()
            gt_concepts = gt_pairs[image_key].upper()

            # Split concept string into concept array
            # Manage empty concept lists
            if gt_concepts.strip() == '':
                gt_concepts = []
            else:
                gt_concepts = gt_concepts.split(';')

            if candidate_concepts.strip() == '':
                candidate_concepts = []
            else:
                candidate_concepts = candidate_concepts.split(';')

            # Manage empty GT concepts (ignore in evaluation)
            if len(gt_concepts) == 0:
                max_score -= 1
            # Normal evaluation
            else:
                # Concepts stats
                total_concepts += len(gt_concepts)

                # Global set of concepts
                all_concepts = sorted(list(set(gt_concepts + candidate_concepts)))

                # Calculate F1 score for the current concepts
                y_true = [int(concept in gt_concepts) for concept in all_concepts]
                y_pred = [int(concept in candidate_concepts) for concept in all_concepts]

                f1score = f1_score(y_true, y_pred, average='binary')

                # Increase calculated score
                current_score += f1score

            # Concepts stats
            nb_concepts = str(len(gt_concepts))
            if nb_concepts not in concepts_distrib:
                concepts_distrib[nb_concepts] = 1
            else:
                concepts_distrib[nb_concepts] += 1

            if len(gt_concepts) > max_concepts:
                max_concepts = len(gt_concepts)

            if len(gt_concepts) < min_concepts:
                min_concepts = len(gt_concepts)

        mean_f1_score = current_score / max_score
        return mean_f1_score

    def train_generator(self, images, labels, batch_size, num_tags):
        random.seed(42)
        num_of_batches = ceil(len(images) / batch_size)
        while True:
            lists = list(zip(images, labels))
            random.shuffle(lists)
            images, labels = zip(*lists)
            for batch in range(num_of_batches):
                if len(images) - (batch * batch_size) < batch_size:
                    current_batch_size = len(images) - (batch * batch_size)
                else:
                    current_batch_size = batch_size
                batch_features = np.zeros((current_batch_size, 224, 224, 3))
                batch_labels = np.zeros((current_batch_size, num_tags))

                for i in range(batch_size):
                    index = (batch * batch_size) + i
                    if index < len(images):
                        batch_features[i] = images[index]
                        batch_labels[i] = labels[index]
                yield batch_features, batch_labels

    def val_generator(self, images, labels, batch_size, num_tags):
        random.seed(42)
        num_of_batches = ceil(len(images) / batch_size)
        while True:
            lists = list(zip(images, labels))
            random.shuffle(lists)
            images, labels = zip(*lists)
            for batch in range(num_of_batches):
                if len(images) - (batch * batch_size) < batch_size:
                    current_batch_size = len(images) - (batch * batch_size)
                else:
                    current_batch_size = batch_size
                batch_features = np.zeros((current_batch_size, 224, 224, 3))
                batch_labels = np.zeros((current_batch_size, num_tags))

                for i in range(batch_size):
                    index = (batch * batch_size) + i
                    if index < len(images):
                        batch_features[i] = images[index]
                        batch_labels[i] = labels[index]
                yield batch_features, batch_labels

    # The number of available concepts
    def chexnet_model(self, num_tags):
        my_init = glorot_uniform(seed=42)
        base_model = DenseNet121(weights='imagenet', include_top=True)
        x = base_model.get_layer("avg_pool").output
        concept_outputs = Dense(num_tags, activation="sigmoid", name="concept_outputs", kernel_initializer=my_init)(x)
        model = Model(inputs=base_model.input, outputs=concept_outputs)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"])

        return model

    def chexnet(self, num_tags, batch_size=2, epochs=2):
        model = self.chexnet_model(num_tags)
        # add early stopping
        early_stopping = EarlyStopping(monitor="val_loss", patience=3, mode="auto", restore_best_weights=True)
        # save best model
        checkpoint = ModelCheckpoint("chexnet_checkpoint.hdf5", monitor="val_loss", save_best_only=True, mode="auto")
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, mode="min")
        # train the model
        x_train, y_train = self.load_data(self.train_data, self.train_concepts)
        x_val, y_val = self.load_data(self.val_data, self.train_concepts)
        history = model.fit_generator(
            self.train_generator(x_train, y_train, batch_size, num_tags),
            steps_per_epoch=ceil(len(x_train) / batch_size),
            epochs=epochs,
            verbose=2,
            callbacks=[early_stopping, checkpoint, reduce_lr],
            validation_data=self.val_generator(x_val, y_val, batch_size, num_tags),
            validation_steps=ceil(len(x_val) / batch_size))

        path = '/home/mary/Documents/Projects/dc/iu_xray/tags.json'
        self._create_dataset(path)
        x, y = self.load_data(self.train_data, self.train_concepts)
        self.chexnet_model(len(self.train_concepts))

path = '/home/mary/Documents/Projects/dc/iu_xray/'
ch = Chexnet(path+'tags.json', path+'iu_xray_images/', 'results_tag')
ch.chexnet()