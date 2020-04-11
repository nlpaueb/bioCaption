import numpy as np
import pandas as pd
import re


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
    dataframe_data["split_captions"] = dataframe_data["captions"]\
        .apply(lambda x: _bioclean(x).split(' '))
    return dataframe_data


def load_data(data_dir):
    data = pd.read_csv(data_dir, sep="\t",
                       names=["image_ids", "captions"],
                       header=None)
    data['img_ids_list'] = data.image_ids.apply(lambda x: x.split(','))
    data = data[['img_ids_list', 'captions']].copy()
    return data


