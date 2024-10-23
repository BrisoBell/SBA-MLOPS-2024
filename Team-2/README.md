# Chest Cancer Classification using MLflow

## Overview

This project implements an end-to-end machine learning pipeline for chest cancer classification using convolutional neural networks (CNN). The goal is to accurately classify chest cancer conditions based on medical imaging data, while employing MLOps practices to ensure reproducibility and efficiency throughout the development and deployment phases.

## Project Structure

- **Workflows**: Contains step-by-step processes for data ingestion, model training, and evaluation.
- **Configuration Files**: 
  - `config.yaml`: Holds the configuration details for various pipeline components.
  - `params.yaml`: Contains model hyperparameters and other adjustable parameters.
- **Pipeline**:
  - `dvc.yaml`: Defines the stages of the ML pipeline, managed by DVC.
  - `main.py`: Main script to trigger the data pipeline and model training.

## Tools and Technologies

- **MLflow**: Used for experiment tracking, model logging, and versioning.
- **DVC**: Lightweight tool for pipeline orchestration and data version control.
- **Python**: Core programming language for model development and implementation.
- **TensorFlow**: Framework used for building and training the CNN model.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/chest-cancer-classification.git
   cd chest-cancer-classification
2. Install the required dependencies
pip install -r requirements.txt

3. Initialize DVC
dvc init

4. Set up MLflow tracking via DagsHub
export MLFLOW_TRACKING_URI=https://dagshub.com/Adithya/chest-Disease-Classification-MLflow-DVC.mlflow
export MLFLOW_TRACKING_USERNAME=Adithya B V
export MLFLOW_TRACKING_PASSWORD=password

5. Run the pipeline
dvc repro

6. Launch the MLflow UI
mlflow ui
