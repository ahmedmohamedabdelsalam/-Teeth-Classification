import numpy as np
import tensorflow as tf
from PIL import Image
import io
import logging
from app.model_loader import model_loader
from app import config

def preprocess_image(image_bytes):
    """
    Process image bytes: Resize, normalize, and convert to tensor.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # Match ImageDataGenerator default: NEAREST interpolation
    img = img.resize(config.IMG_SIZE, Image.Resampling.NEAREST)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def get_prediction(image_bytes):
    """
    Predict teeth condition from image bytes.
    """
    try:
        model = model_loader.load_model()
        processed_image = preprocess_image(image_bytes)
        
        predictions = model.predict(processed_image)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        predicted_class = config.CLASS_LABELS[class_idx]
        
        # Prepare all scores for visualization
        all_scores = {label: float(prob) for label, prob in zip(config.CLASS_LABELS, predictions[0])}

        
        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "all_scores": all_scores
        }
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise e
