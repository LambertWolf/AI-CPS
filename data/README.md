# Data Folder

This folder contains all CSV files used across scraping, training, evaluation, and inference for the S-Bahn delay prediction project.

## Core datasets

- `joint_data_collection.csv`  
  Combined/base dataset used for feature engineering and model training.  
  Expected columns include: `timestamp`, `stop_id`, `delay_minutes`.

- `training_data_ann_improved.csv`  
  Training split for the improved ANN pipeline (feature-aligned with the ANN model inputs).

- `test_data_ann_improved.csv`  
  Test split for the improved ANN pipeline (used for evaluation and final KPIs).

## Scraper support

- `s7_delays.csv`  
  Helper file for the scraper. It stores collected S7 delay-related information (e.g., runs/trips and timing references) so the scraping process knows which trains are currently operating and which time windows to query/track.

## Notes
- CSV files are kept free of secrets and credentials.
- If filenames or paths change, update the corresponding references in the scraper/training/inference scripts (or your config).

