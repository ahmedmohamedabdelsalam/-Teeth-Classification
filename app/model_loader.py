import tensorflow as tf
import os
import logging

from app import config

class ModelLoader:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def load_model(self):
        if self._model is None:
            model_path = config.MODEL_PATH
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            logging.info(f"Loading model from {model_path}...")
            self._model = tf.keras.models.load_model(model_path)
            logging.info("Model loaded successfully.")
        return self._model

model_loader = ModelLoader()
