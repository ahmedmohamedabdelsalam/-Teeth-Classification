import os
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Add project root to path for config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import config

def build_model(num_classes):
    """
    Builds a MobileNetV2 based transfer learning model.
    """
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(config.IMG_SIZE[0], config.IMG_SIZE[1], 3)
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train():
    # Paths (adjust based on where the user places the data)
    data_dir = os.path.join(config.BASE_DIR, "data", "raw")
    train_dir = os.path.join(data_dir, "Training")
    val_dir = os.path.join(data_dir, "Validation")
    
    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found at {train_dir}")
        print("Please ensure your dataset is organized as data/raw/Training and data/raw/Validation")
        return

    # Data Augmentation & Generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True, # Oral images can often be flipped vertically
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=config.IMG_SIZE,
        batch_size=32,
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=config.IMG_SIZE,
        batch_size=32,
        class_mode='categorical'
    )

    # Build and Train
    model = build_model(len(config.CLASS_LABELS))
    
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(config.MODEL_PATH, save_best_only=True, verbose=1),
        ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    ]
    
    print("Starting Training (Retraining from Scratch)...")
    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    # Optional: Unfreeze and fine-tune
    print("Retraining Complete. Model saved to:", config.MODEL_PATH)

if __name__ == "__main__":
    train()
