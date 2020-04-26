import os
import sys
import json
from sklearn.metrics import f1_score
from bioCaption.configuration import get_logger


class TagsEvaluation:
    logger = get_logger()

    def __init__(self, gold_dir=None, results_dir=None, gold_data={}, result_data={}):
        self.golds_dir = gold_dir
        self.results_dir = results_dir
        self.gt_pairs = gold_data
        self.candidate_pairs = result_data

    def _load_data(self):
        with open(self.golds_dir) as json_file:
            self.gt_pairs = json.load(json_file)
        with open(self.results_dir) as json_file:
            self.candidate_pairs = json.load(json_file)

    def evaluate_f1(self):
        if self.golds_dir is not None and \
                self.results_dir is not None:
            self._load_data()

        # Concept stats
        min_concepts = sys.maxsize
        max_concepts = 0
        total_concepts = 0
        concepts_distrib = {}

        # Define max score and current score
        max_score = len(self.gt_pairs)
        current_score = 0

        # Check there are the same number of pairs between candidate and ground truth
        if len(self.candidate_pairs) != len(self.gt_pairs):
            print('ERROR : Candidate does not contain the same number of entries as the ground truth!')
            exit(1)

        # Evaluate each candidate concept list against the ground truth
        for image_key in self.candidate_pairs:
            # Get candidate and GT concepts
            candidate_concepts = self.candidate_pairs[image_key]
            gt_concepts = self.gt_pairs[image_key]

            # Split concept string into concept array
            # Manage empty concept lists
            if len(gt_concepts) == 0:
                gt_concepts = []

            if len(candidate_concepts) != 0:
                candidate_concepts = candidate_concepts.split(';')
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
                print(f1score)

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


f1 = TagsEvaluation(gold_dir='/home/mary/Documents/Projects/bioCaption/iu_xray/tags.json', results_dir='/home/mary/Documents/Projects/bioCaption/results_knn.json')
f1.evaluate_f1()
