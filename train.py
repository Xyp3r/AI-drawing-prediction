# This script trains an improved deep learning model for the
# Quick Draw! doodle recognition task.
#
# Changes from the previous version:
# - Uses a larger set of categories for more comprehensive training.
# - Increases the number of samples per category to improve accuracy.
# - The model architecture is made more complex with additional layers.
# - Adds a Dropout layer to prevent overfitting.
#
# To run this script:
# 1. Ensure you have the required libraries installed:
#    pip install tensorflow numpy matplotlib
# 2. Run the script from your terminal:
#    python train.py
#
# The script will download the dataset, train the model, and save it to
# 'quick_draw_model.h5', overwriting the old one.

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import urllib.request

# --- Configuration for Improved Training ---
# Using a larger, more diverse set of categories.
# This will significantly improve the model's ability to generalize.
CATEGORIES = [
    'cat', 'dog', 'tree', 'car', 'house', 'bicycle', 'sun', 'star',
    'face', 'flower', 'mountain', 'bus', 'airplane', 'cloud', 'ocean'
]
# Increasing the number of samples per category for more robust training.
NUM_SAMPLES_PER_CATEGORY = 20000 
IMAGE_SIZE = 28
MODEL_FILE = 'quick_draw_model.h5'

def download_and_load_data(categories, num_samples):
    """
    Downloads and loads a subset of the Quick, Draw! dataset.
    This function downloads .npy files from Google's public bucket.
    """
    print("Downloading and loading data...")
    all_images = []
    all_labels = []

    for i, category in enumerate(categories):
        # The URL for the .npy files
        url = f'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy'
        
        # Download the file
        file_path = f'{category}.npy'
        if not os.path.exists(file_path):
            print(f"Downloading {category}.npy...")
            try:
                urllib.request.urlretrieve(url, file_path)
            except urllib.error.HTTPError as e:
                print(f"Error downloading {category}: {e}. Skipping this category.")
                continue
        
        # Load the data
        data = np.load(file_path, allow_pickle=True)
        
        # Take a subset and normalize
        images = data[:num_samples] / 255.0
        images = images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
        labels = np.full(num_samples, i)

        all_images.append(images)
        all_labels.append(labels)

    # Concatenate all data
    if all_images and all_labels:
        images = np.concatenate(all_images, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        print("Data loaded successfully.")
        return images, labels
    else:
        print("No data was loaded. Please check category names and internet connection.")
        return np.array([]), np.array([])


def build_model(num_categories):
    """
    Builds an improved Convolutional Neural Network (CNN) model.
    - Added an extra convolutional layer for deeper feature extraction.
    - Added a Dropout layer to reduce overfitting.
    - Increased the number of neurons in the dense layers.
    """
    model = Sequential([
        Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_categories, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("Improved model built successfully.")
    model.summary()
    return model

def train_model(model, x_train, y_train):
    """
    Trains the model on the prepared data.
    - The number of epochs is increased for more thorough training.
    """
    # Convert labels to one-hot encoding
    y_train_one_hot = to_categorical(y_train, num_classes=len(CATEGORIES))
    
    # Define a callback to save the best model during training
    checkpoint = ModelCheckpoint(MODEL_FILE, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    
    print("Starting improved model training...")
    history = model.fit(x_train, y_train_one_hot,
                        epochs=20,  # Increased epochs for better accuracy
                        batch_size=256,
                        validation_split=0.2,
                        callbacks=[checkpoint])
    print("Model training complete.")
    return history

def main():
    """
    Main function to orchestrate the training process.
    """
    images, labels = download_and_load_data(CATEGORIES, NUM_SAMPLES_PER_CATEGORY)
    
    if len(images) > 0:
        model = build_model(len(CATEGORIES))
        train_model(model, images, labels)
        print(f"Model saved to {MODEL_FILE}")
    else:
        print("Training cannot proceed without data.")

if __name__ == '__main__':
    main()
