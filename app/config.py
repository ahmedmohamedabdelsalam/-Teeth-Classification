import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Model Configuration
MODEL_NAME = "teeth_model.keras"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)

# Inference Configuration
IMG_SIZE = (224, 224)
CLASS_LABELS = ["CaS", "CoS", "Gum", "MC", "OC", "OLP", "OT"]

# Server Configuration
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 7860))
DEBUG = True
