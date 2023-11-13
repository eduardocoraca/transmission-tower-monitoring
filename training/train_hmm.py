import sys

sys.path.append(".")

from classes.feature import PCAFeatureReducer, WPTFeatureExtractor
from classes.loader.loader import Loader

PATH_TO_TRAIN_DATASET = r"./raw/sequence_2/"
PATH_TO_VALID_DATASET = r"./raw/sequence_1/"
CABLES = [1, 2, 3, 4]

feature_params = {"family": "db4", "level": 5}
reducer_params = {"dimension": 2}

params = feature_params.copy()
params.update(reducer_params)

feature_extractor = WPTFeatureExtractor(
    normalize=True, axis_to_combine=["y", "z"], **feature_params
)
loader = Loader(PATH_TO_VALID_DATASET, feature_extractor)
dfs_valid = loader.load_from_folder()

reducer = PCAFeatureReducer(**params)

for cable, df in dfs_valid.items():
    num_cable = int(cable[-1])
    df["reduced"] = df["features"].apply(
        lambda x: reducer.transform(x.reshape(1, -1), num_cable)
    )
