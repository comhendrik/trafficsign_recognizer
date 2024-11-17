import cv2
import os
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D

import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Function to load images from a directory
def load_images_from_folder(folder, target_size=(32, 32)):
    images = []
    labels = []
    class_names = os.listdir(folder)
    class_names = [class_name for class_name in class_names if class_name.isdigit()]  # Keep only numeric folder names# Get subfolder names as class labels
    class_names = sorted(class_names, key=lambda x: int(x))# Get subfolder names as class labels
    labelSub = 0
    for label, class_name in enumerate(class_names):
        print(label)
        print(class_name)


# os specific
        if class_name == ".DS_Store":
            labelSub = 1
            continue


        class_folder = os.path.join(folder, class_name)
        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)
            img = cv2.imread(img_path)  # Load image
            if img is None: continue
            img = cv2.resize(img, target_size)  # Resize
            images.append(img)
            labels.append(label-labelSub)
    return np.array(images), np.array(labels)

# Load training images
train_images, train_labels = load_images_from_folder("Images")
print("Train Images Shape:", train_images.shape)
print("Train Labels Shape:", train_labels.shape)

import tensorflow as tf
from tensorflow.keras import layers, models

# Modell-Architektur
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input, MaxPool2D, BatchNormalization
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical


X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.3, random_state=42, shuffle=True)

X_train = X_train/255
X_val = X_val/255

print("X_train.shape", X_train.shape)
print("X_valid.shape", X_val.shape)
print("y_train.shape", y_train.shape)
print("y_valid.shape", y_val.shape)

print(X_train[0])



y_train = to_categorical(y_train, num_classes=43)
y_val = to_categorical(y_val, num_classes=43)

print(y_train.shape)
print(y_val.shape)


model = Sequential([
    Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(32,32,3)),
    Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    BatchNormalization(axis=-1),

    Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    BatchNormalization(axis=-1),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(rate=0.5),

    Dense(43, activation='softmax')
])


lr = 0.001
epochs = 30

opt = Adam(learning_rate=lr, decay=lr / (epochs * 0.5))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest"
)

history = model.fit(aug.flow(X_train, y_train, batch_size=32), epochs=epochs, validation_data=(X_val, y_val))
# Modell speichern
model.save('traffic_sign_model.keras')
