import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Set Streamlit Page Config
st.set_page_config(
    page_title="🦷 Teeth Classification AI",
    page_icon="🦷",
    layout="wide",
)

# Load the trained model
MODEL_PATH = "mobilenetv2_teeth_classifier.keras"
model = load_model(MODEL_PATH)

# Define class labels
class_labels = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (224, 224))  # Resize to match model input shape
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Debugging: Print the shape to confirm resizing worked
    print(f"Processed Image Shape: {img.shape}")  # Should be (1, 224, 224, 3)
    
    return img



# Function to display sample images
def show_sample_images():
    sample_dir = "./Teeth_Dataset/Training"
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))

    for i, class_name in enumerate(os.listdir(sample_dir)[:5]):  # Show 5 classes
        class_path = os.path.join(sample_dir, class_name)
        img_name = os.listdir(class_path)[0]  # Take first image
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)
        axes[i].set_title(class_name)
        axes[i].axis("off")

    st.pyplot(fig)

# Streamlit UI
st.title("🦷 AI-Powered Teeth Classification")
st.write("Upload an image of teeth to classify its condition.")

# Sidebar
st.sidebar.header("ℹ️ About This App")
st.sidebar.info(
    """
    - **Model:** MobileNetV2 Fine-tuned for Teeth Classification
    - **Classes:** CaS, CoS, Gum, MC, OC, OLP, OT
    - **Image Size:** 256x256 pixels
    - **Framework:** TensorFlow & Streamlit
    """
)
st.sidebar.subheader("🔍 Sample Images")
show_sample_images()

# File uploader
uploaded_file = st.file_uploader("📂 Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Display uploaded image
    st.image(img, caption="📸 Uploaded Image", use_column_width=True)

    # Preprocess and predict
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    
    # Get top prediction
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # Display results
    st.subheader(f"🦷 Predicted Condition: **{class_labels[predicted_class]}**")
    st.write(f"Confidence: **{confidence:.2%}**")

    # Show class probabilities
    st.write("### 📊 Prediction Probabilities:")
    prob_df = {class_labels[i]: f"{prediction[0][i]:.2%}" for i in range(len(class_labels))}
    st.json(prob_df)

# Footer
st.sidebar.text("🚀 Created by Ahmed Abdelsalam")

