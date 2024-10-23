import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Define the model path
        model_path = os.path.join("artifacts", "training", "model.keras")
        print(f"Loading model from: {model_path}")

        # Check if the model file exists
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found at {model_path}. Please ensure it exists.")

        # Load the model
        model = load_model(model_path)

        # Load and preprocess the test image
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Normalize the image (if you did this during training)
        test_image /= 255.0  # Scale pixel values to [0, 1]

        # Make prediction
        raw_predictions = model.predict(test_image)
        print("Raw predictions:", raw_predictions)  # Print the raw prediction output
        result = np.argmax(raw_predictions, axis=1)

        # Interpret the prediction
        if result[0] == 1:
            prediction = 'Normal'
            return [{"image": prediction}]
        else:
            prediction = 'Adenocarcinoma Cancer'
            return [{"image": prediction}]