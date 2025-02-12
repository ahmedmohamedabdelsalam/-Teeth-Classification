# Teeth Classification from Scratch

## Overview
This project implements a deep learning model for teeth classification using TensorFlow and Keras. The model is built from scratch and trained on a custom dataset.

## Project Repository
GitHub Repository: [Teeth Classification](https://github.com/ahmedmohamedabdelsalam/-Teeth-Classification)

## Dataset
The dataset consists of images categorized into different classes. It is divided into three sets:
- **Training Set**: Used to train the model
- **Validation Set**: Used to validate performance during training
- **Testing Set**: Used to evaluate final model performance

Dataset is stored in a ZIP file named `Teeth_Dataset.zip` and is extracted automatically when running the script.

## Model Architecture
The deep learning model is a Convolutional Neural Network (CNN) built using Keras with the following layers:
- **Convolutional Layers**: Extract spatial features from images
- **Max Pooling Layers**: Reduce dimensionality and computation
- **Batch Normalization**: Improve training stability
- **Dropout Layers**: Prevent overfitting
- **Dense Layers**: Perform final classification

## Requirements
To run this project, install the required dependencies:
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

## Training the Model
Run the following command to train the model:
```bash
python train.py
```
- The model is trained for **50 epochs** with an **Adam optimizer** and **categorical cross-entropy loss**.
- The best model is saved as `best_model.h5` using ModelCheckpoint.

## Evaluation & Results
After training, the model is evaluated on the test set, generating:
- **Accuracy & Loss Plots**
- **Confusion Matrix**
- **Classification Report**

Run the following command to evaluate the trained model:
```bash
python evaluate.py
```

## Model Performance
The model achieves:
- Competitive **accuracy** on test data
- A well-balanced confusion matrix and classification report

## Future Improvements
- Implement data augmentation techniques
- Experiment with different architectures (ResNet, VGG, etc.)
- Optimize hyperparameters for better accuracy

## License
This project is open-source under the MIT License.

