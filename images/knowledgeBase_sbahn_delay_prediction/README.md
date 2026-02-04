# knowledgeBase - S-Bahn Delay Prediction

## Ownership
- Project Ownership: Lambert Wolf, Jakob Varenbud

## Context
- This image was created as part of the course "M. Grum: Advanced AI-based Application Systems" by the "Junior Chair for Business Information Science, esp. AI-based Application Systems" at University of Potsdam

## AI Model Characterization
This container provides a trained TensorFlow/Keras Artificial Neural Network (ANN) for S-Bahn delay prediction.

### Model Architecture
- **Type**: Deep Neural Network (DNN) with 3 hidden layers
- **Framework**: TensorFlow/Keras
- **Input Features**:
  - 10 temporal features (hour, weekday, cyclical encodings, rush hour indicators)
  - 3 lag features (previous delays at the station)
  - 2 station statistics (mean delay, std delay)
  - One-hot encoded station identifiers
- **Architecture**:
  - Hidden Layer 1: 64 neurons (ReLU activation) + Dropout (0.2)
  - Hidden Layer 2: 32 neurons (ReLU activation) + Dropout (0.2)
  - Hidden Layer 3: 16 neurons (ReLU activation)
  - Output Layer: 1 neuron (linear activation for regression)
- **Output**: Predicted delay in minutes

### Artifacts Included
- `ann_improved_model.keras`: Trained Keras model
- `ann_improved_scaler.pkl`: StandardScaler for feature normalization
- `ann_improved_feature_names.pkl`: List of feature names in correct order

### Training Details
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam (learning_rate=0.001)
- Regularization: Dropout layers, Early Stopping
- Data Split: 80% training, 20% test/validation

## License
- We are committing to the "AGPL-3.0 license"

## Usage
This container provides trained models at `/tmp/knowledgeBase/` for AI-based delay prediction.

To use with docker-compose, mount the external volume `ai_system:/tmp`.
