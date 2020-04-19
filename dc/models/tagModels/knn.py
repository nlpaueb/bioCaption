import os
import sys
import numpy as np
from tqdm import tqdm
from collections import Counter
from dc.data.data_functions import create_tag_dataset
from keras.applications.densenet import DenseNet121
from keras.models import Model
import keras.applications.densenet as densenet
from keras.preprocessing import image
from sklearn.metrics import f1_score

sys.path.append("..")  # Adds higher directory to python modules path.


class Knn:

    def __init__(self, data_dir, images_dir, results_dir):
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
        self.data_dir = data_dir
        self.ids = []
        self.raw = []
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def _load_data(self):
        self.train_data, self.test_data, self.val_data, _, _\
            = create_tag_dataset(self.data_dir)

    def compute_image_embedding(self, image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        img_vec= image.img_to_array(img)
        img_vec = np.expand_dims(img_vec, axis=0)
        img_vec = densenet.preprocess_input(img_vec)
        return img_vec

    def create_vectors_for_image_set(self, data, img_path, vector_extraction_model):
        train_images_vec = {}
        print("Extracting embeddings for all train images...")
        for train_image in tqdm(data.keys()):
            image_path = os.path.join(img_path, train_image)
            x = self.compute_image_embedding(image_path)
            vec = vector_extraction_model.predict(x).transpose().flatten()
            train_images_vec[train_image] = vec
            print("Got embeddings for train images.")
        return train_images_vec



    # F1 evaluation function downloaded from the competition
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

        # Evaluate each candidate concept list against the ground truth
        i = 0
        for image_key in candidate_pairs:

            # Get candidate and GT concepts
            candidate_concepts = candidate_pairs[image_key]
            gt_concepts = gt_pairs[image_key]

            # Split concept string into concept array
            # Manage empty concept lists
            if len(gt_concepts) != 0:
                gt_concepts = gt_concepts[0].split(';')

            if len(candidate_concepts) != 0:
                candidate_concepts = candidate_concepts[0].split(';')
            else:
                candidate_concepts = []

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

    def knn(self):
        self._load_data()
        base_model = DenseNet121(weights='imagenet', include_top=True)
        vector_extraction_model = \
            Model(inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output)
        print("Calculating visual embeddings from train images")
        train_images_vec = {}
        print("Extracting embeddings for all train images...")
        for train_image in tqdm(self.train_data.keys()):
            image_path = os.path.join(self.images_dir, train_image)
            x = self.compute_image_embedding(image_path)
            vec = vector_extraction_model.predict(x).transpose().flatten()
            train_images_vec[train_image] = vec
            print("Got embeddings for train images.")

        # save IDs and raw image vectors seperately but aligned
        self.ids = [i for i in train_images_vec]
        self.raw = np.array([train_images_vec[i] for i in train_images_vec])
        # normalize image vectors to avoid normalized cosine and use dot
        self.raw = self.raw / np.array([np.sum(self.raw, 1)] * self.raw.shape[1]).transpose()

        # measure the similarity of each val img embedding with all train img embeddings
        images_sims = {}
        for val_image in tqdm(self.val_data.keys()):
            image_path = os.path.join(self.images_dir, val_image)
            x = self.compute_image_embedding(image_path)
            vec = vector_extraction_model.predict(x).transpose().flatten()
            vec = vec / np.sum(vec)
            # clone to do efficient mat mul dot
            test_mat = np.array([vec] * self.raw.shape[0])
            sims = np.sum(test_mat * self.raw, 1)
            # save the similarities array for every test image
            images_sims[val_image] = sims

        print("Found similarities of validation images.")
        max_score, best_k = self.tune_k(images_sims)
        print("Found best score on val:", max_score, "for k =", best_k)
        self.test_knn(best_k)

    def tune_k(self, images_sims):
        # tune k at validation data
        best_k = 1
        max_score = 0  # max score is initially zero
        for k in tqdm(range(1, 201)):  # search k from 1 to 200
            val_results = {}  # store validation images and their predicted concepts
            for image_sim in images_sims:  # for each val image get all similarities of that image with train images
                topk = np.argsort(images_sims[image_sim])[-k:]  # get the k most similar images
                concepts_list = []  # store the concepts for that image
                sum_concepts = 0  # store total num of concepts in the k images
                for index in topk:  # for each similar image update the concept list (of the test image in question)
                    concepts = self.train_data[self.ids[index]]
                    sum_concepts += len(concepts)
                    for concept in concepts:
                        concepts_list.append(concept)
                frequent_concepts = Counter(concepts_list).most_common(round(sum_concepts / k))
                val_results[image_sim] = ";".join(
                    f[0] for f in frequent_concepts)  # process to match competition evaluation
            # evaluate k-nn for this k and if better, update the max score
            score = self.evaluate_f1(self.val_data, val_results)
            if score > max_score:
                max_score = score
                best_k = k
        return max_score, best_k

    def test_knn(self, best_k):
        base_model = DenseNet121(weights='imagenet', include_top=True)
        vector_extraction_model = \
            Model(inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output)
        sim_test_results = {}
        test_images = self.test_data
        for i, test_image in tqdm(enumerate(test_images)):
            image_path = os.path.join(self.images_dir, test_image)
            img = image.load_img(image_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = densenet.preprocess_input(x)
            vec = vector_extraction_model.predict(x).transpose().flatten()
            vec = vec / np.sum(vec)
            # clone to do efficient mat mul dot
            test_mat = np.array([vec] * self.raw.shape[0])
            sims = np.sum(test_mat * self.raw, 1)
            topk = np.argsort(sims)[-best_k:]
            concepts_list = []
            sum_concepts = 0
            for index in topk:
                concepts = self.train_data[self.ids[index]]
                sum_concepts += len(concepts)
                for concept in concepts:
                    concepts_list.append(concept)
            frequent_concepts = Counter(concepts_list).most_common(round(sum_concepts / best_k))
            sim_test_results[test_image] = ";".join(f[0] for f in frequent_concepts)
        print("Saving test results...")
        with open(os.path.join(self.results_dir,"results_knn.csv"), 'w') as output_file:
            for result in sim_test_results:
                output_file.write(result + "\t" + sim_test_results[result])
                output_file.write("\n")



