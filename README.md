# Transmission Tower Monitoring

Research code for monitoring guyed transmission towers using vibration measurements and cable tension data. This repository contains the data-processing and machine-learning experiments developed for a PhD project.

The monitoring data is organized around four cable accelerometers and tension sensors. Accelerations are measured on the x, y, and z axes, while the x axis is used as the input acceleration in the raw measurement records.

## Scientific workflow

The project follows this experimental pipeline:

1. Parse raw text exports containing acceleration, tension, temperature, and sample timing.
2. Extract features such as:
   - frequency response functions (FRFs);
   - power spectral densities (PSDs);
   - statistical waveform features;
   - wavelet packet energy and entropy maps.
3. Combine cable axes and select a frequency range of interest.
4. Train latent representation models to monitor changes in cable behavior:
   - variational autoencoder with reconstruction, KL, and cable-distance regularization;
   - autoencoder with cosine-distance regularization;
   - VAE-based HMM pipeline for sequence analysis.
5. Compare embeddings against a healthy baseline and analyze reconstruction error or latent distance over time.
6. Log experiments, hyperparameters, metrics, and models with Comet.

The three main time-ordered measurement sequences are defined in `training/constants.py`.

## Repository structure

```text
classes/
  feature/       Statistical feature Pydantic models
  loader/        Data loading utilities
  peak/          Detected peak records
  sample/        Raw measurement inputs and cable/sample Pydantic models
  signal/        Signal, spectrum, FRF, and wavelet packet models
lib/
  hmm_utils.py   HMM construction and transition-matrix helpers
  utils.py       JSON and folder loading utilities
  wpt_utils.py   Wavelet packet loading and processing
processing/
  *_extraction.py Raw measurement extraction entry points
training/
  constants.py   Sequences, sampling rate, and cable definitions
  dataset.py     Dataset construction and tensor normalization
  models/        PyTorch Lightning autoencoder and VAE models
  evaluate.py    Embedding and reconstruction evaluation
  training_pipeline.py
  training_pipeline_v2.py
  train_vae_hmm_pipe.py
  main_training_script_v1.py
  main_training_script_v2.py
  main_vae_hmm_pipe.py
notebooks/       Exploratory and experimental notebooks
examples/        Small model demonstrations
```

## Setup

This project uses Python 3.11 or newer and `uv` for dependency management.

```powershell
uv sync
```

For a direct Python/virtual environment setup:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

GPU training requires a CUDA-compatible PyTorch installation and is configured as the primary accelerator in the training pipelines.

## Environment variables

Copy the example environment file:

```powershell
Copy-Item .env.example .env
```

Set the Comet API key in `.env`:

```dotenv
COMET_API_KEY=your_comet_api_key
COMET_WORKSPACE=your_workspace
COMET_PROJECT_VAE=your_vae_project
COMET_PROJECT_COSINE=your_cosine_project
COMET_PROJECT_VAE_HMM=your_hmm_project
COMET_MODEL_NAME=your_model_name
COMET_MODEL_VERSION=your_model_version
```

The `.env` file is ignored by Git. `.env.example` contains only variable names and contains no credentials.

## Data layout

Raw and processed data is intentionally excluded from Git. The scripts expect the main training sequences under:

```text
data/
  wpts/
    sequence_1_db35/
    sequence_2_db35/
    sequence_3_db35/
```

The raw data consists of text measurement files containing one cable/sample record per file. Extracted WPT, PSD, FRF, statistical, and embedding outputs are written to JSON or other ignored data formats under `data/`.

Set `RAW_DATA_PATH` to the local directory containing the raw measurement files.

The `processing/` modules can be used to convert raw measurement files into Pydantic-backed data structures and JSON outputs. Notebook data, TensorBoard/Lightning logs, checkpoints, images, archives, CSVs, and model weights are generated artifacts and are ignored by Git.

## Training

The active training entry points are the scripts under `training/`.

Typical usage:

```powershell
uv run python training/main_training_script_v1.py
uv run python training/main_training_script_v2.py
uv run python training/main_vae_hmm_pipe.py
```

These scripts load the three sequences, run hyperparameter combinations, train PyTorch Lightning models, and log experiments through Comet.

The main pipelines are:

- `training/training_pipeline.py`: regularized VAE with reconstruction, KL, and cable-distance losses.
- `training/training_pipeline_v2.py`: cosine-regularized autoencoder.
- `training/train_vae_hmm_pipe.py`: VAE embeddings followed by sequence/HMM analysis.

Training parameters are currently defined inside the pipeline scripts and include batch size, learning rate, latent dimension, hidden dimension, number of layers, frequency range, wavelet level range, and regularization weights.

## Data extraction

The extraction modules use raw text measurement files as input:

- `processing/wpt_extraction.py`
- `processing/psd_extraction.py`
- `processing/frf_extraction.py`
- `processing/statistical_features_extraction.py`
- `processing/peak_extraction.py`

Each module provides lower-level functions that operate on `LMSFile` objects and `extract_*_and_save` functions for batch conversion.

## Exploratory notebooks

The `notebooks/` directory contains exploratory analysis, feature engineering, WPT/PSD/FRF visualization, HMM experiments, and model inference. Notebooks are research artifacts and may include superseded approaches or partial experiments.

## Experimental logging

Comet is used for:

- hyperparameter sweeps;
- training and validation metrics;
- model checkpoints;
- embedding and distance plots;
- correlation metrics.

Experiments require `COMET_API_KEY` and can be run with the scripts or notebooks that instantiate `CometLogger`.
