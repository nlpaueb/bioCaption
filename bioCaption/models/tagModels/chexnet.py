import os
import sys
import numpy as np
import json
import math
from tqdm import tqdm
from keras.applications.densenet import preprocess_input
from keras.layers import Dense
from keras.initializers import glorot_uniform
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.applications.densenet import DenseNet121
from keras.models import Model
from keras.preprocessing import image
from bioCaption.models.tagModels.tag_models_evaluation import TagsEvaluation

np.random.seed(42)
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
        self.split_ratio = split_ratio
        self._create_dataset(self.data_dir)
        self.model = None
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
        # creates images and labels
        return np.array(x_data), np.array(y_data)

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

    def train_generator(self, images, labels, batch_size, num_tags):
        np.random.seed(42)
        num_of_batches = math.ceil(len(images) / batch_size)
        while True:
            lists = list(zip(images, labels))
            np.random.shuffle(lists)
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
        np.random.seed(42)
        num_of_batches = math.ceil(len(images) / batch_size)
        while True:
            lists = list(zip(images, labels))
            np.random.shuffle(lists)
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
        self.model = model

    def tune_decision_threshold(self):
        pass

    def chexnet(self, batch_size=2, epochs=2, model_path=None):
        num_tags = len(self.train_concepts)
        print(num_tags)
        self.chexnet_model(num_tags)
        # add early stopping
        early_stopping = EarlyStopping(monitor="val_loss", patience=3, mode="auto", restore_best_weights=True)
        # save best model
        checkpoint = ModelCheckpoint(os.path.join(model_path, "chexnet_checkpoint.hdf5"),
                                     monitor="val_loss",
                                     save_best_only=True,
                                     mode="auto")
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, mode="min")
        # train the model
        x_train, y_train = self.load_data(self.train_data, self.train_concepts)
        x_val, y_val = self.load_data(self.val_data, self.train_concepts)
        history = self.model.fit_generator(
            self.train_generator(x_train, y_train, batch_size, num_tags),
            steps_per_epoch=math.ceil(len(x_train) / batch_size),
            epochs=epochs,
            verbose=2,
            callbacks=[early_stopping, checkpoint, reduce_lr],
            validation_data=self.val_generator(x_val, y_val, batch_size, num_tags),
            validation_steps=math.ceil(len(x_val) / batch_size)
        )

    def load_trained_model(self, weights_path):
        self.chexnet_model(len(self.train_concepts))
        self.model.load_weights(weights_path)

    def chexnet_test(self, decision_threshold=0.5, model_path=None):
        if model_path is not None:
            self.load_trained_model(model_path)
        print("Loading test images...")
        test_images, img_ids = self.load_images_ids(self.test_data)

        # get predictions for test
        test_predictions = self.model.predict(test_images, batch_size=5, verbose=1)
        print("Got predictions for dev set")
        results = {}
        for i in range(len(test_predictions)):
            concepts_pred = []
            for j in range(len(self.train_concepts)):
                if test_predictions[i, j] >= decision_threshold:
                    concepts_pred.append(self.train_concepts[j])
            results[img_ids[i]] = ";".join(concepts_pred)
        # evaluate results
        test_score = self.f1_evaluation(self.test_data, results)
        print("The F1 score on test set is: ", test_score)
        # save results
        with open(os.path.join(self.results_dir, 'chexnet_results.json'), 'w') as json_file:
            json.dump(results, json_file)
