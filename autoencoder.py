"""
 Simple Autoencoder for Monocular Depth Estimation
 @author: Adam Santos
"""
import os

import numpy as np
from keras.constraints import maxnorm
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Reshape
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import load_model
import cv2 as cv

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass
print("Training autoencoder...")
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.33)
# Create the model
model = Sequential()
# model.add(Flatten(input_shape = input_shape))
# model.add(Dense(512))
model.add(Conv2D(8, kernel_size=3, padding='same', activation='relu', input_shape=(284,75,1)))
model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.5))
# model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.5))
# model.add(Conv2D(32, kernel_size=5, padding='same', activation='relu', kernel_constraint=maxnorm(3)))
# model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
# model.add(Dropout(0.5))
# model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dropout(0.5))
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(21300, activation='relu'))
model.add(Reshape((284,75,1)))
# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy')
model.summary()

history = model.fit(train_images, train_depths, batch_size=16, epochs=60,
                    validation_data=(test_images, test_depths))

