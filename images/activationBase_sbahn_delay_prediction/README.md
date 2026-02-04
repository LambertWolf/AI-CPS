# activationBase - S-Bahn Delay Prediction

## Ownership
- Project Ownership: Lambert Wolf, Jakob Varenbud

## Context
- This image was created as part of the course "M. Grum: Advanced AI-based Application Systems" by the "Junior Chair for Business Information Science, esp. AI-based Application Systems" at University of Potsdam

## Data Origin
- The data was scraped from the exclusive VBB (Verkehrsverbund Berlin-Brandenburg) API
- Contains a single activation data entry for model inference testing
- Data includes engineered features: temporal features, lag features, station statistics, and one-hot encoded stations

## License
- We are committing to the "AGPL-3.0 license"

## Usage
This container provides activation data at `/tmp/activationBase/activation_data.csv` for testing AI model inference.

To use with docker-compose, mount the external volume `ai_system:/tmp`.
