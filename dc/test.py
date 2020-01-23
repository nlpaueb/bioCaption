from dc.download.download_datasets import *
from dc.models.baselines import Baselines
from dc.evaluation.evaluation_metrics import Evaluation




#download_iu_xray()
print("I'm here")
baselines = Baselines("iu_xray/train_images.tsv", "iu_xray/test_images.tsv", "iu_xray/iu_xray_images", "results")
print("Done")
baselines.most_frequent_word_in_captions(10)
evaluation = Evaluation("results/most_frequent_word_results.tsv", "iu_xray/test_images.tsv")
evaluation.compute_ms_coco()
