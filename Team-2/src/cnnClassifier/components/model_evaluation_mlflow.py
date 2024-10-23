import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
import os
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories, save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        print(f"MLflow Tracking URI: {self.config.mlflow_uri}")

    def _valid_generator(self):
        print("Creating validation data generator...")
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )
        print("Validation data generator created.")

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        print(f"Loading model from path: {path}")
        model = tf.keras.models.load_model(path)
        print("Model loaded successfully.")
        return model

    def evaluation(self):
        print("Starting model evaluation...")
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        print(f"Evaluation completed with loss: {self.score[0]} and accuracy: {self.score[1]}")
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        print(f"Saving scores to scores.json: {scores}")
        save_json(path=Path("scores.json"), data=scores)
        print("Scores saved successfully.")

    def log_into_mlflow(self):
        print("Logging into MLflow...")
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run() as run:
            print(f"MLflow run started with ID: {run.info.run_id}")
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})
            print("Parameters and metrics logged.")

            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
                print(f"Model registered as 'VGG16Model' on MLflow.")
            else:
                mlflow.keras.log_model(self.model, "model")
                print("Model logged locally to MLflow.")
        print("MLflow logging completed.")
