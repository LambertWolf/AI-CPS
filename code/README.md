# Code (S-Bahn Delay Prediction)

This folder contains the source code for our AI-CPS course project.  
We predict S-Bahn delays in minutes for a given stop and timestamp using two approaches: a baseline **OLS regression** and an improved **ANN**.

## Folder structure

### `Ann_code/`
Neural network (TensorFlow/Keras) training pipelines:
- `ann.py` — baseline ANN model
- `ann_improved.py` — improved ANN with additional feature engineering (e.g., lag features, station statistics, extended cyclical time encoding)

### `Ols_code/`
Baseline regression model:
- `train_ols_model.py` — trains and exports an OLS model for the same prediction task

### `scraper/`
Data collection pipeline for S7 delays:
- `main.py` — main entry point for scraping
- `config.py` — scraper configuration
- `vbb_api.py` — API interaction layer
- `collected_data.jsonl` — raw collected records (JSON Lines)
- `seen_keys.json` — deduplication state to avoid re-scraping
- `s7_delays_*.csv` — helper CSVs used by the scraper to track relevant S7 runs/time windows
- `test_scraper.py` — basic scraper tests

## Task overview
- **Scrape** S7 delay-related data → store raw records → build CSV datasets
- **Train** OLS and ANN models on the prepared data
- **Evaluate** using MAE and R² and generate plots/metrics
- **Apply** the trained models to the example activation data in the Docker scenarios

## License
AGPL-3.0

