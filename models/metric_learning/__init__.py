import sys

sys.path.append("..")

from yaml import Loader, load

from .model_classes import DataLoader, TripletDataset
from .model_utils import create_triplets_idx, get_proba_per_sample
from .models import FeatureReducerScikit, MLPFeatureReducer

with open("../models/metric_learning/params.yaml", "r") as f:
    PARAMS = load(f, Loader)
