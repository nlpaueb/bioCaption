# bioCaption - Diagnostic Captioning
Based on [A Survey on Biomedical Image Captioning](https://www.aclweb.org/anthology/W19-1803).

> V. Kougia, J. Pavlopoulos and I Androutsopoulos, "A Survey on Biomedical Image Captioning". 
Proceedings of the Workshop on Shortcomings in Vision and Language of the Annual Conference 
of the North American Chapter of the Association for Computational Linguistics (NAACL-HLT 2019), Minneapolis, USA, 2019.

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
