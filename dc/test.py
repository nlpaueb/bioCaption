from dc.download.download_datasets import *
from dc.models.baselines import Baselines
from dc.evaluation import evaluation_metrics




download_iu_xray()
baselines = Baselines()
baselines.one_nn()