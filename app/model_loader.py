import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
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

    def _build_architecture(self):
        """
        Rebuilds the exact architecture used in training.
        """
        base_model = MobileNetV2(
            weights=None, # Architecture only 
            include_top=False, 
            input_shape=(config.IMG_SIZE[0], config.IMG_SIZE[1], 3)
        )
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.4)(x)
        predictions = Dense(len(config.CLASS_LABELS), activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        return model

    def load_model(self):
        if self._model is None:
            weights_path = config.MODEL_PATH
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Weights file not found at {weights_path}")
            
            logging.info(f"Building architecture and loading weights from {weights_path}...")
            self._model = self._build_architecture()
            self._model.load_weights(weights_path)
            logging.info("Model weights loaded successfully.")
        return self._model

model_loader = ModelLoader()
