# S-Bahn Delay Prediction (AI-CPS Project)

This repository contains our course project for predicting public transport delays (in minutes) for a given stop and timestamp.  
We build two models: a baseline **OLS regression** and an improved **ANN (neural network)** with feature engineering.

 We have trained the model with data we collected ourselves via the VBB API. However, it is also conceivable to collect (presumably less accurate) data via the [gtfs standard](https://gtfs.org/). 

## What it does
- Predicts `delay_minutes` for each (`stop_id`, `timestamp`)
- Uses engineered features such as:
  - time features (hour/weekday + cyclical sin/cos)
  - rush-hour / weekend indicators
  - station one-hot encoding
  - lag features (previous delays per station)
  - station statistics (mean/std delay)

## Project structure (high level)
- `results/` – trained models, scalers/encoders, plots, and exported artifacts
- `scripts/` or `src/` – training, evaluation, and inference code (ANN + OLS)
- `docker/` (or `images/`) – Dockerfiles for Subgoal 6/7 images and compose setups

## Key outputs
- ANN model: `ann_improved_model_new.keras`
- OLS model: `currentOlsSolution.pkl`
- Activation data: `activation_data.csv`
- Plots: training history, predicted-vs-actual, feature importance

## Docker (Subgoals 6 & 7)
We publish three Docker Hub images and provide docker-compose files for running:
1) **[knowledgeBase](https://hub.docker.com/repository/docker/maggusrulez/knowledgebase_sbahn_delay_prediction)** – model artifacts under `/tmp/knowledgeBase/`
2) **[activationBase](https://hub.docker.com/repository/docker/maggusrulez/activationbase_sbahn_delay_prediction/general)** – activation data under `/tmp/activationBase/`
3) **[codeBase](https://hub.docker.com/repository/docker/maggusrulez/codebase_sbahn_delay_prediction/general)** – applies ANN or OLS to the activation data and writes prediction outputs into the shared `/tmp` volume

## License
AGPL-3.0

