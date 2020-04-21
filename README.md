# DC - Diagnostic Captioning
Based on [A Survey on Biomedical Image Captioning](https://www.aclweb.org/anthology/W19-1803).

> V. Kougia, J. Pavlopoulos and I Androutsopoulos, "A Survey on Biomedical Image Captioning". 
Proceedings of the Workshop on Shortcomings in Vision and Language of the Annual Conference 
of the North American Chapter of the Association for Computational Linguistics (NAACL-HLT 2019), Minneapolis, USA, 2019.

### Install
To install the package locally
```
git clone https://github.com/nlpaueb/dc.git
cd dc
python3 setup.py sdist
pip install dist/dc-0.1.tar.gz 
```

### How to use

```python
from dc.data.downloads import DownloadData
from dc.models.baselines import Baselines
from dc.evaluation.evaluation import Evaluation

downloads = DownloadData()
# download the iu_xray dataset in the current directory
downloads.download_iu_xray()

baselines = Baselines('iu_xray/train_images.tsv','iu_xray/test_images.tsv','iu_xray/iu_xray_images/','results')
baselines.most_frequent_word_in_captions()

evaluation = Evaluation('iu_xray/test_images.tsv', 'results/most_frequent_word_results.tsv')
evaluation.compute_WMD()

```

### Providing your own dataset.
You'll need to provide two tsv files, one for training and one for testing.
The dataset needs to have the following syntax:

#### For caption generation.
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

#### For medical image tagging.
```json
{
  "imgid1": ["tag1", "tag2", "tag3"],
  "imgid2": ["tag1", "tag2"],
  "imgid3": ["tag1"]....
}
```
- Please note:
    - Each img_id corresponds to an actual image name stored separately
into an image's folder.