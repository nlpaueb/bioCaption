import os
import re
import json
import math
import numpy as np
import pandas as pd


def average_embedding(embeddings_matrix):
    return np.average(embeddings_matrix, axis=0)


# Clean for BioASQ
def _bioclean(caption):
    return re.sub('[.,?;*!%^&_+():-\[\]{}]',
                  '',
                  caption.replace('"', '')
                  .replace('/', '')
                  .replace('\\', '')
                  .replace("'", '')
                  .strip()
                  .lower()).split()


def get_list_of_words_per_caption(dataframe_data):
    dataframe_data["split_captions"] = dataframe_data["caption"]\
        .apply(lambda x: _bioclean(x))
    return dataframe_data


def load_data(data_dir):
    data = pd.read_csv(data_dir, sep="\t",
                       names=["image_ids", "caption"],
                       header=None)
    data['img_ids_list'] = data.image_ids.apply(lambda x: x.split(','))
    return data


def save_results(results_dictionary, results_dir, file_name):
    # Save test results to tsv file
    df = pd.DataFrame.from_dict(results_dictionary, orient="index")
    df.to_csv(os.path.join(results_dir, file_name+"tsv"), sep="\t", header=False)


def download_bio_embeddings(path):
    os.system("wget "+"-P "+path+" https://archive.org/download/pubmed2018_w2v_200D.tar/pubmed2018_w2v_200D.tar.gz")
    # Unzip word embeddings
    os.system("tar xvzf pubmed2018_w2v_200D.tar.gz")
    os.system("rm  pubmed2018_w2v_200D.tar.gz")


def create_tag_dataset(path, split_dataset=[0.6, 0.1, 0.3]):
    with open(path) as json_file:
        data = json.load(json_file)
        keys = list(data.keys())
        train_pointer = math.ceil(split_dataset[0]*len(keys))
        test_pointer = math.ceil(split_dataset[1]*len(keys))
        val_pointer = math.ceil(split_dataset[2]*len(keys))
        train_keys = keys[:train_pointer]
        test_keys = keys[train_pointer:train_pointer+test_pointer]
        val_keys = keys[train_pointer+test_pointer:val_pointer+train_pointer+test_pointer]
        train = {}
        test = {}
        val = {}
        train_concepts = []
        val_concepts = []
        for key in train_keys:
            train[key] = data[key]
            train_concepts.extend(data[key])
        for key in test_keys:
            test[key] = data[key]
        for key in val_keys:
            val[key] = data[key]
            val_concepts.extend(data[key])
    return train, test, val, train_concepts, val_concepts



