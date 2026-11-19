import sys
from training.comet_config import get_comet_config
from dotenv import load_dotenv

load_dotenv()

sys.path.append(".")
sys.path.append("..")

import matplotlib

matplotlib.use("Agg")
from pytorch_lightning.loggers import CometLogger
from pathlib import Path
from lib.wpt_utils import load_wavelets
from training_pipeline_v2 import run_pipeline
from training.objects import PipelineOutput
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from itertools import product
from pydantic import BaseModel
import itertools
import scipy.stats


COMET_CONFIG = get_comet_config("COMET_PROJECT_COSINE")

SEQ_1_PATH = Path("./data/wpts/sequence_1_db35")
SEQ_2_PATH = Path("./data/wpts/sequence_2_db35")
SEQ_3_PATH = Path("./data/wpts/sequence_3_db35")

data_seq_1 = load_wavelets(SEQ_1_PATH)
data_seq_2 = load_wavelets(SEQ_2_PATH)
# data_test = load_wavelets(SEQ_3_PATH)

data_train, data_valid = data_seq_2, data_seq_1

param_combinations = list(
    product(
        beta := [1e-2, 1e-1, 1, 1e1],
        latent_dim := [2, 4, 8, 16, 32],
        f_min := [7],
        bandwidth := [113],
        min_level := [3],
        max_level := [5, 6, 7],
    )
)

# filter out parameter with min_level >= max_level

fixed_params = {
    "normalize": True,
    "batch_size": 64,
    "num_epochs": 200,
    "lr": 1e-3,
    "hidden_dim": 16,
    "num_layers": 3,
    "activation": "relu",
    "fs": 256,
}

for k, combination in enumerate(param_combinations):
    (
        beta,
        latent_dim,
        f_min,
        bandwidth,
        min_level,
        max_level,
    ) = combination

    variable_params = {
        "latent_dim": latent_dim,
        "f_min": f_min,
        "bandwidth": bandwidth,
        "min_level": min_level,
        "max_level": max_level,
        "beta": beta,
        "num_comb": k,
    }
    params = {**fixed_params, **variable_params}
    logger = CometLogger(
        api_key=COMET_CONFIG["api_key"],
        project_name=COMET_CONFIG["project_name"],
        workspace=COMET_CONFIG["workspace"],
    )
    logger.log_hyperparams(params)
    try:
        output = run_pipeline(data_train, data_valid, params, logger)
        # log_metrics(output, logger)
        # time.sleep(2)
    except Exception as e:
        print(e)
        continue
    logger.experiment.end()
