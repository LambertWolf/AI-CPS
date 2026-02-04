# Docker Images

This folder contains the Docker image definitions used in our project and course subgoals.

## Overview

### `knowledgeBase_sbahn_delay_prediction/`
**Purpose:** Provides the trained model artifacts (ANN + OLS) and documentation inside the container.  
**Expected paths (when used with the shared `/tmp` volume):**
- `/tmp/knowledgeBase/` (model files + README)

---

### `activationBase_sbahn_delay_prediction/`
**Purpose:** Provides the example activation input data used to apply the trained models.  
**Expected paths:**
- `/tmp/activationBase/` (e.g., `activation_data.csv`, `activation_data_ols.csv`)

---

### `codeBase_sbahn_delay_prediction/`
**Purpose:** Applies the trained models to the activation data (inference).  
**Reads from:**
- `/tmp/knowledgeBase/` (model artifacts)
- `/tmp/activationBase/` (activation input CSV)  
**Writes results to:**
- `/tmp/...` in the shared volume (e.g., prediction result files)

---

### `learningBase_sbahn_delay_prediction/`
**Purpose:** Training/evaluation image used to train models and generate plots/artifacts in a reproducible environment.  
Typically used to create/refresh exported models, scalers/encoders, and documentation plots.

## Notes
- Those images are designed to be used together via docker-compose with an external volume:
  `ai_system:/tmp`
- The shared `/tmp` volume acts as the exchange layer between provisioning images and the inference image.

