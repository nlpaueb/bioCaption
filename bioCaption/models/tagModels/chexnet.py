import json
import math
import os
import sys
import numpy as np
import pandas as pd
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.initializers import glorot_uniform
from keras.layers import Dense
from keras.models import Model
from keras.preprocessing import image
from tqdm import tqdm

from bioCaption.models.tagModels.tag_models_evaluation import TagsEvaluation

np.random.seed(42)
sys.path.append("..")  # Adds higher directory to python modules path.


class Chexnet:

    def __init__(self, data_path, images_dir, results_dir, batch_size=10,
                 split_ratio=[0.5, 0.4, 0.1]):
        """
        :param data_path: Path to json training data file.
        :param images_dir: : The folder in the dataset that contains the images
         with the form: "[dataset_name]_images".
        :param split_ratio: List of floats that represents the ratios of the
        dataset to be used as train, test and validation. The default value is
        [0.5, 0.4, 0.1]. The ratios must always sum to 1.0.
        :param results_dir: The folder in which to save the results file. If it
        doesn't exist, it get created automatically in the current working
        directory.
        """
        self.images_dir = images_dir
        self.results_dir = results_dir
        self.train_data = {}
        self.test_data = {}
        self.val_data = {}
        self.train_concepts = []
        self.val_concepts = []
        self.data_path = data_path
        self.split_ratio = split_ratio
        self._create_dataset(self.data_path)
        self.model = None
        self.batch_size = batch_size
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def f1_evaluation(self, gt_pairs, candidate_pairs):
        f1 = TagsEvaluation(gold_data=gt_pairs,
                            result_data=candidate_pairs)
        return f1.evaluate_f1()

    def _create_dataset(self, path):
        with open(path) as json_file:
            data = json.load(json_file)
            keys = list(data.keys())
            train_pointer = math.ceil(self.split_ratio[0] * len(keys))
            test_pointer = math.ceil(self.split_ratio[1] * len(keys))
            val_pointer = math.ceil(self.split_ratio[2] * len(keys))
            train_keys = keys[:train_pointer]
            test_keys = keys[train_pointer:train_pointer + test_pointer]
            val_keys = keys[
                       train_pointer + test_pointer:val_pointer + train_pointer + test_pointer]
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
        # creates images and labels
        return np.array(x_data), np.array(y_data)

    def load_concepts(self, concept_path):
        self.train_concepts = [line.rstrip('\n') for line in open(concept_path)]

    def load_images_ids(self, data):
        x_images = []
        x_images_ids = []
        for img_id in tqdm(data):
            image_path = os.path.join(self.images_dir, img_id)
            img = image.load_img(image_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = preprocess_input(x)
            x_images.append(x)
            x_images_ids.append(img_id)
        return np.array(x_images), x_images_ids

    def train_generator(self, images, labels, num_tags):
        np.random.seed(42)
        num_of_batches = math.ceil(len(images) / self.batch_size)
        while True:
            lists = list(zip(images, labels))
            np.random.shuffle(lists)
            images, labels = zip(*lists)
            for batch in range(num_of_batches):
                if len(images) - (batch * self.batch_size) < self.batch_size:
                    current_batch_size = len(images) - (batch * self.batch_size)
                else:
                    current_batch_size = self.batch_size
                batch_features = np.zeros((current_batch_size, 224, 224, 3))
                batch_labels = np.zeros((current_batch_size, num_tags))

                for i in range(self.batch_size):
                    index = (batch * self.batch_size) + i
                    if index < len(images):
                        batch_features[i] = images[index]
                        batch_labels[i] = labels[index]
                yield batch_features, batch_labels

    def val_generator(self, images, labels, num_tags):
        np.random.seed(42)
        num_of_batches = math.ceil(len(images) / self.batch_size)
        while True:
            lists = list(zip(images, labels))
            np.random.shuffle(lists)
            images, labels = zip(*lists)
            for batch in range(num_of_batches):
                if len(images) - (batch * self.batch_size) < self.batch_size:
                    current_batch_size = len(images) - (batch * self.batch_size)
                else:
                    current_batch_size = self.batch_size
                batch_features = np.zeros((current_batch_size, 224, 224, 3))
                batch_labels = np.zeros((current_batch_size, num_tags))

                for i in range(self.batch_size):
                    index = (batch * self.batch_size) + i
                    if index < len(images):
                        batch_features[i] = images[index]
                        batch_labels[i] = labels[index]
                yield batch_features, batch_labels

    # The number of available concepts
    def chexnet_model(self, num_tags):
        my_init = glorot_uniform(seed=42)
        base_model = DenseNet121(weights='imagenet', include_top=True)
        x = base_model.get_layer("avg_pool").output
        concept_outputs = Dense(num_tags, activation="sigmoid",
                                name="concept_outputs",
                                kernel_initializer=my_init)(x)
        model = Model(inputs=base_model.input, outputs=concept_outputs)
        model.compile(optimizer="adam", loss="binary_crossentropy",
                      metrics=["binary_accuracy"])
        self.model = model

    def chexnet(self, batch_size=100, epochs=2, model_path=None):
        """
        Train the chexnet model for medical image tagging.
        :param batch_size: size of batches. Default is 100.
        :param epochs: Training epochs. Default is 2.
        :param model_path: Path were the best model will be stored.
        """
        num_tags = len(self.train_concepts)
        self.chexnet_model(num_tags)
        # add early stopping
        early_stopping = EarlyStopping(monitor="val_loss", patience=3,
                                       mode="auto", restore_best_weights=True)
        # save best model
        checkpoint = ModelCheckpoint(
            os.path.join(model_path, "chexnet_checkpoint.hdf5"),
            monitor="val_loss",
            save_best_only=True,
            mode="auto")
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                      patience=1, verbose=1, mode="min")
        # train the model
        x_train, y_train = self.load_data(self.train_data, self.train_concepts)
        x_val, y_val = self.load_data(self.val_data, self.val_concepts)
        history = self.model.fit_generator(
            self.train_generator(x_train, y_train, self.batch_size, num_tags),
            steps_per_epoch=math.ceil(len(x_train) / self.batch_size),
            epochs=epochs,
            verbose=2,
            callbacks=[early_stopping, checkpoint, reduce_lr],
            validation_data=self.val_generator(x_val, y_val, self.batch_size,
                                               num_tags),
            validation_steps=math.ceil(len(x_val) / self.batch_size)
        )

    def load_trained_model(self, weights_path):
        self.chexnet_model(len(self.train_concepts))
        self.model.load_weights(weights_path)

    def chexnet_test(self, decision_threshold=0.5, model_path=None, concepts_path=None, save_results=True):
        """
        :param save_results: Set true to export results to json.
        :param concepts_path: Path to concepts
        :param decision_threshold: Decision threshold for chexnet classifier.
        Default value is 0.5.
        :param model_path: path where the model is stored. By default its value
        is None and the chexnet_test is trying to use the value that is stored in
        the 'model' attribute of the object. In case model_path is not None
        then it replaces the value of the 'model' attribute into the
        model that is loaded from the path
        """
        if concepts_path is not None:
            self.load_concepts(concepts_path)
        if model_path is not None:
            self.load_trained_model(model_path)
        print("Loading test images...")
        test_images, img_ids = self.load_images_ids(self.test_data)

        # get predictions for test
        test_predictions = self.model.predict(test_images, batch_size=self.batch_size,
                                              verbose=1)
        print("Got predictions.")
        results_concepts = []
        results_json = {}
        for i in range(len(test_predictions)):
            concepts_pred = []
            for j in range(len(self.train_concepts)):
                if test_predictions[i, j] >= decision_threshold:
                    concepts_pred.append(self.train_concepts[j])
            if save_results:
                results_json[img_ids[i]] = ";".join(concepts_pred)
            results_concepts.append(concepts_pred)

        # evaluate results
        #test_score = self.f1_evaluation(self.test_data, results)
        #print("The F1 score on test set is: ", test_score)
        # save results
        if save_results:
            with open(os.path.join(self.results_dir, 'chexnet_results.json'),
                      'w') as json_file:
                json.dump(results_json, json_file)
        return img_ids, results_concepts
    
    def chexnet_ensemple(self, checkpoints_directory, concepts_directory, checkpoints_threshold=None,
                         detailed_results=True):
        """
        :param checkpoints_directory: Directory with chexnet model checkpoints.
        :param concepts_directory: : Directory with concepts files for each checkpoint.
        :param checkpoints_threshold: Dictionary with the decition threshold for each dictionary.
        :param detailed_results: If True the final csv will contain results for each checkpoint else
        only the intersection of results for each image is being written..
        """
        if checkpoints_threshold is None:
            checkpoints_threshold = {
                "1_DRAN": 0.34,
                "2_DRAN": 0.7,
                "1_DRCO": 0.14,
                "2_DRCO": 0.08,
                "1_DRCT": 0.73,
                "2_DRCT": 0.5,
                "1_DRMR": 0.28,
                "2_DRMR": 0.98,
                "1_DRPE": 0.3,
                "2_DRPE": 0.23,
                "1_DRUS": 0.18,
                "2_DRUS": 0.22,
                "1_DRXR": 0.45,
                "2_DRXR": 0.86
            }
        ensemble_models_results = {}
        checkpoints = os.listdir(checkpoints_directory)
        for checkpoint in checkpoints:
            model_path = os.path.join(checkpoints_directory, checkpoint)
            threshold = checkpoints_threshold[checkpoint.replace("_tagCXN_checkpoint.hdf5", "")]
            concepts_path = os.path.join(concepts_directory, checkpoint.replace("_tagCXN_checkpoint.hdf5", "_concepts.txt"))
            img_ids, results = self.chexnet_test(decision_threshold=threshold, model_path=model_path, concepts_path=concepts_path)
            ensemble_models_results[checkpoint] = results
        df = pd.DataFrame.from_dict(ensemble_models_results)
        df['models_intesection'] = df[df.columns].apply(lambda x: list(set.intersection(*map(set, list(x)))), axis=1)
        df.insert(0, 'image_ids', img_ids)
        if detailed_results:
            df.to_csv('chexnet_results_detailed.csv')
        else:
            df[["image_ids", "models_intesection"]].to_csv('chexnet_results.csv')
