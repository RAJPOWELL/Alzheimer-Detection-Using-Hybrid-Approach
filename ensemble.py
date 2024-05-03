# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

import os
from distutils.dir_util import copy_tree, remove_tree

from PIL import Image
from random import randint

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix

import tensorflow_addons as tfa
from keras.utils.vis_utils import plot_model
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization, GlobalAveragePooling2D


print("TensorFlow Version:", tf.__version__)

# Data Pre-Processing
base_dir = "/kaggle/input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/"
root_dir = "./"
test_dir = base_dir + "test/"
train_dir = base_dir + "train/"
work_dir = root_dir + "dataset/"

if os.path.exists(work_dir):
    remove_tree(work_dir)

os.mkdir(work_dir)
copy_tree(train_dir, work_dir)
copy_tree(test_dir, work_dir)
print("Working Directory Contents:", os.listdir(work_dir))
WORK_DIR = './dataset/'

CLASSES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

IMG_SIZE = 176
IMAGE_SIZE = [176, 176]
DIM = (IMG_SIZE, IMG_SIZE)

# Performing Image Augmentation to have more data samples
ZOOM = [.99, 1.01]
BRIGHT_RANGE = [0.8, 1.2]
HORZ_FLIP = True
FILL_MODE = "constant"
DATA_FORMAT = "channels_last"

work_dr = IDG(rescale=1./255, brightness_range=BRIGHT_RANGE, zoom_range=ZOOM, data_format=DATA_FORMAT, fill_mode=FILL_MODE, horizontal_flip=HORZ_FLIP)

train_data_gen = work_dr.flow_from_directory(directory=WORK_DIR, target_size=DIM, batch_size=6500, shuffle=False)

# Define custom CNN model
def custom_cnn_model():
    model = Sequential([
        Input(shape=(*IMAGE_SIZE, 3)),
        Conv2D(16, 3, activation='relu', padding='same'),
        Conv2D(16, 3, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.2),
        Conv2D(32, 3, activation='relu', padding='same'),
        Conv2D(32, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(),
        Dropout(0.2),
        Conv2D(64, 3, activation='relu', padding='same'),
        Conv2D(64, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(4, activation='softmax')
    ], name="custom_cnn_model")
    return model

# Define InceptionV3 model
def inception_model():
    inception_base = InceptionV3(input_shape=(176, 176, 3), include_top=False, weights="imagenet")
    for layer in inception_base.layers:
        layer.trainable = False

    model = Sequential([
        inception_base,
        Dropout(0.5),
        GlobalAveragePooling2D(),
        Flatten(),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        BatchNormalization(),
        Dense(4, activation='softmax')
    ], name="inception_model")
    return model

# Train the custom CNN model
custom_cnn = custom_cnn_model()
custom_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
custom_cnn.fit(train_data_gen, epochs=10)

# Train the InceptionV3 model
inception = inception_model()
inception.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
inception.fit(train_data_gen, epochs=10)

# Evaluate both models
test_data, test_labels = train_data_gen.next()
custom_cnn_accuracy = custom_cnn.evaluate(test_data, test_labels)
inception_accuracy = inception.evaluate(test_data, test_labels)

print("Custom CNN Model Accuracy:", custom_cnn_accuracy)
print("InceptionV3 Model Accuracy:", inception_accuracy)

# Combine predictions from both models
custom_cnn_predictions = custom_cnn.predict(test_data)
inception_predictions = inception.predict(test_data)
combined_predictions = np.argmax(custom_cnn_predictions + inception_predictions, axis=1)

# Evaluate ensemble performance
ensemble_accuracy = accuracy_score(np.argmax(test_labels, axis=1), combined_predictions)
ensemble_balanced_accuracy = balanced_accuracy_score(np.argmax(test_labels, axis=1), combined_predictions)
ensemble_precision, ensemble_recall, ensemble_f1_score, _ = precision_recall_fscore_support(np.argmax(test_labels, axis=1), combined_predictions, average='weighted')
ensemble_confusion_matrix = confusion_matrix(np.argmax(test_labels, axis=1), combined_predictions)

print("Ensemble Model Accuracy:", ensemble_accuracy)
print("Ensemble Model Balanced Accuracy:", ensemble_balanced_accuracy)
print("Ensemble Model Precision:", ensemble_precision)
print("Ensemble Model Recall:", ensemble_recall)
print("Ensemble Model F1-Score:", ensemble_f1_score)
print("Ensemble Model Confusion Matrix:\n", ensemble_confusion_matrix)

# Save the ensemble model
ensemble_model_dir = "ensemble_model.h5"
tf.keras.models.save_model(ensemble, ensemble_model_dir)
print("Ensemble Model saved as:", ensemble_model_dir)
