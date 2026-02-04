# Scenarios

This folder contains ready-to-run docker-compose scenarios.  
Each scenario pulls the required images from Docker Hub and runs the full pipeline using the shared external volume `ai_system` mounted to `/tmp`.

## Structure

### `apply_ann_solution/docker-compose.yml`
Runs the **ANN application** workflow:
1) cleans `/tmp` in the shared volume  
2) provisions `/tmp/knowledgeBase/` (model artifacts)  
3) provisions `/tmp/activationBase/` (activation input CSV)  
4) runs the codeBase container to apply the ANN model and write prediction outputs into the shared `/tmp` volume

### `apply_ols_solution/docker-compose.yml`
Runs the **OLS application** workflow:
1) cleans `/tmp` in the shared volume  
2) provisions `/tmp/knowledgeBase/` (model artifacts)  
3) provisions `/tmp/activationBase/` (activation input CSV)  
4) runs the codeBase container to apply the OLS model and write prediction outputs into the shared `/tmp` volume

## Notes
- Both scenarios depend on the external Docker volume `ai_system`. Create it once if needed:
  `docker volume create ai_system`
- After execution, results and intermediate files can be inspected inside the volume under `/tmp`.

