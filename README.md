# bioCaption - Diagnostic Captioning

The datasets and captioning models are described in detail in ‘A Survey on Biomedical Image Captioning’.
The tagging models were used for the participation of AUEB NLP Group in the ImageCLEFmed 2019 Concept Detection task, where they achieved the best performance. More detail can be found in the systems description paper and in the Best of Labs track for the 2019 ImageCLEF Concept Detection task of the 2020 CLEF conference proceeding (Kougia et al. 2020).
In the ImageCLEFmed 2020 Concept Detection task, AUEB NLP Group achieved again the best performance and was ranked 1st, 2nd and 6th. We provide the pretrained tagging systems that were ranked first, so they can be downloaded and used for medical image tagging. More details about these systems can be found in the systems description paper (Karatzas et al., 2020).
# Datasets

We provide scripts for downloading IU X-ray and Peir Gross. Both datasets can be used for captioning, while IU X-ray can also be used for tagging.


#### IU X-ray
IU X-ray (Demner-Fushman et al., 2015) is provided by the Open Access Biomedical Image Search Engine (OpenI) and it consists of radiology examinations (cases). Each case corresponds to one or more images, one radiology report and two sets of tags. The reports have 4 sections: Comparison, Indication, Findings and Impression, of which Findings and Impression can be used for captioning. There are two types of tags: the MTI tags, which are extracted from the text by the Medical Text Indexer and the manual tags, which are assigned by two trained coders. In total, there are 3,955 reports with 7,470 frontal and lateral X-rays.

#### Peir Gross
This dataset was first used for captioning by Jing et al. (2018). It consists of photographs of medical incidents provided by the Pathology Education Informational Resource (PEIR) digital library for use in medical education. There are 7,443 images extracted from the Gross collections of 21 PEIR pathology sub-categories, each associated with a caption. Each caption is a single descriptive sentence.


# Tagging

#### Mean@k-NN
Mean@k-NN uses an image retrieval method to assign tags to images by computing cosine similarities between image embeddings. Given a test image, this system retrieves the k most similar training images and their tags. The most frequent tags of the retrieved images are assigned to the test image. The average number of tags per retrieved image is calculated to decide how many tags will be assigned to the test image. The value of k is tuned on validation data.


####  TagCXN (also called ConceptCXN)
This system performs multi-label classification. It uses the DenseNet-121 CNN as an image encoder. The image embeddings are extracted from the last average pooling layer of the encoder and passed through a dense layer that produces a probability for each tag. At test time, first, the threshold that is used to assign tags to an image is tuned on validation data. Then, the tags that have a probability that exceeds the resulting threshold are assigned to each test image.


# Captioning

#### Frequency Baseline
This baseline uses the most frequent words in the training set to form a caption. This caption is then assigned to every test image. The length of the caption is the average length of the training captions.

#### Nearest Neighbour Baseline (1-NN)
1-NN uses cosine similarity to find the training image that is the most similar to a test image and assigns the retrieved training caption to the test image. The images are encoded by a CNN and their image embedding extracted from a layer of the encoder is used.

#### References
V. Kougia, J. Pavlopoulos and I. Androutsopoulos, "A Survey on Biomedical Image Captioning". Proceedings of the Workshop on Shortcomings in Vision and Language of the Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL-HLT 2019), Minneapolis, USA, pp. 26-36, 2019.

V. Kougia, J. Pavlopoulos and I. Androutsopoulos, "AUEB NLP Group at ImageCLEFmed Caption 2019". Working Notes of the Conference and Labs of the Evaluation Forum (CLEF 2019), Lugano, Switzerland, 2019.

V. Kougia, J. Pavlopoulos and I. Androutsopoulos, "Medical Image Tagging by Deep Learning and Retrieval". Experimental IR Meets Multilinguality, Multimodality, and Interaction
Proceedings of the Eleventh International Conference of the CLEF Association (CLEF 2020), Thessaloniki, Greece, 2020.

B. Karatzas, J. Pavlopoulos, V. Kougia and I. Androutsopoulos, "AUEB NLP Group at ImageCLEFmed Caption 2020". Working Notes of the Conference and Labs of the Evaluation Forum (CLEF 2020), Thessaloniki, Greece, 2020.

B. Jing,  P. Xie, E. Xing, “On the Automatic Generation of Medical Imaging Reports”. Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics(Long Papers), Melbourne, Australia, pp. 2577–2586, 2018.

D. Demner-Fushman, M. D. Kohli, M. B. Rosenman,S. E. Shooshan, L. Rodriguez, S. Antani, G. R.Thoma, and C. J. McDonald, “Preparing a collection of radiology examinations for distribution and retrieval”. Journal of the American Medical Informatics Association, 23(2):304–310, 2015.
<br><br><br>
# Practical Guide
### Install
To install the package
```
pip install bioCaption
```

### Mecical Image Captioning
#### How to use
```python
from bioCaption.data.downloads import DownloadData
from bioCaption.models.captionModels.baselines import Baselines
from bioCaption.models.captionModels.caption_models_evaluation import CaptionsEvaluation

downloads = DownloadData()
# download the iu_xray dataset in the current directory
downloads.download_iu_xray()

baselines = Baselines('iu_xray/train_images.tsv','iu_xray/test_images.tsv','iu_xray/iu_xray_images/','results')
baselines.most_frequent_word_in_captions()

evaluation = CaptionsEvaluation('iu_xray/test_images.tsv', 'results/most_frequent_word_results.tsv')

# if the directory "embeddings" does not exits, it will be created
# and the embeddings will be downloaded there.
evaluation.compute_WMD('embeddings/', embeddings_file="pubmed2018_w2v_200D.bin")
evaluation.compute_ms_coco()
```

#### Providing your own dataset.
You'll need to provide two tsv files, one for training and one for testing.
The dataset needs to have the following syntax:

```tsv
img_id_11,img_id_12,img_id13   caption1
img_id21 caption2
img_id31,img_31 caption3
```
- Please note:
    - There are no spaces after each comma.
    - Between the image ids and the caption there's a tab (/t).
    - Each img_id corresponds to an actual image name stored separately
into an image's folder.

#### Results
Results are saved in the 'results' folder, in a tsv file with the form.
```json
{
  "imgid1,imgid2": "caption1",
  "imgid3": "caption2",
  "imgid4,imgid5": "caption3"
}
```
### Medical Image Tagging
#### K-NN
```python
from bioCaption.data.downloads import DownloadData
from bioCaption.models.tagModels.knn import Knn

downloads = DownloadData()
# download the iu_xray dataset in the current directory
downloads.download_iu_xray()

knn = Knn('iu_xray/tags.json', 'iu_xray/iu_xray_images/', 'results_tag')
best_k = knn.knn()
knn.test(best_k)
```

#### cheXNet
```python
from bioCaption.data.downloads import DownloadData
from bioCaption.models.tagModels.chexnet import Chexnet

downloads = DownloadData()
# download the iu_xray dataset in the current directory
downloads.download_iu_xray()

"""
Load data and split the into train, test and
evaluation according to the split ratio.
"""
chexnet = Chexnet('iu_xray/tags.json',
                  'iu_xray/iu_xray_images/',
                  'results_tag',
                   split_ratio = [0.6, 0.3, 0.1])

"""
Train the model and checkpoint to model_path.
""" 
chexnet.chexnet(batch_size=2, epochs=2, model_path='iu_xray')

""" While training, chexnet saves the model into the
instance variable "chexnet.model". If "model_path" is None
the model stored into chexnet.model is used else
there's an attempt to load the model from the "model path"
 """
chexnet.chexnet_test(decision_threshold=0.5, model_path=None)
```

#### Input data file.
```json
{
  "imgid1": ["tag1", "tag2", "tag3"],
  "imgid2": ["tag1", "tag2"],
  "imgid3": ["tag1"]
}
```
- Please note:
    - Each img_id corresponds to an actual image name stored separately
into an image's folder.

#### Results
```json
{
  "imgid1": ["tag1", "tag2", "tag3"],
  "imgid2": ["tag1", "tag2"],
  "imgid3": ["tag1"]
}
```

#### Combining multiple chexnet models.

You can train multiple chexnet models and then combine their decisions during testing with the "chexnet_ensemple" function.
To use it, you need to have all the checkpoints (*.hdf5 files) in a directory alongside another directory which will contain the text files with the concepts (one for each checkpoint). 
Each checkpoint should be named as `[nameCheckpoint1]_tagCXN_checkpoint.hdf5` and its corresponging concept file should be named as `[nameCheckpoint1]_concepts.txt`

```python
from bioCaption.models.tagModels.chexnet import Chexnet

"""
Note the split ratio in the constructor below.
Since we don't want to train a model anymore, rather than only use the chexnet_ensemble,
we want all the data that we load to be used as test. There's no need to assign part
 of the data for training or validation in this case.
"""
chexnet = Chexnet('data/tags.json',
                  'data/images/',
                  'results_dir',
                  batch_size=30,
                  split_ratio=[0.0, 0.0, 1.0])

# Dictionary which contains the decision threshold for each checkpoint.
# It can be left None, then function is using the thresholds for the ImageCLEF 2020 checkpoints.
checkpoints_threshold = {
    "nameCheckpoint1": 0.34,
    "nameCheckpoint2": 0.7,
    "nameCheckpoint3": 0.14
}

chexnet.chexnet_ensemple("data/checkpoints", "data/concepts",
 checkpoints_threshold = checkpoints_threshold, detailed_results=True)
```
