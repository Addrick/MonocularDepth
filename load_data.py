"""
 Loads KITTI dataset from local folders
 Expects single pool of driving images and depth pngs
 and train_test_splits them into two sets
 @author: Adam Santos
"""
import os

import numpy as np
from keras.constraints import maxnorm
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import load_model
import cv2 as cv
import read_depth as rd

# TODO: include the rest of the raw city driving scenes
# currently uses the raw dataset for BW driving images and the annotated depth maps set under 'depth completion' on KITTI website
# TODO: figure out how to use the velodyne .bin files instead of the depth completion dataset
x_data_path = "data/bw_img/"
y_data_path = "data/depth/"
# load data from folders based on folder name:
training_images = []
training_depths = []

images = os.listdir(x_data_path)
depths = os.listdir(y_data_path)
print(len(images))
print(len(depths))

# load images and depth maps into array
for im_name in images:
    im = x_data_path + im_name
    img = cv.imread(im,cv.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv.resize(img, dsize=(284,75))
        training_images.append(img)
for depth in depths:
    dm = x_data_path + depth
    depth_map = rd.depth_read(dm)
    if depth_map is not None:
        depth_map = cv.resize(depth_map, dsize=(284,75))
        training_depths.append(depth_map)

input_shape = img.shape
output_shape = depth_map.shape
output_size = output_shape[0]*output_shape[1]

training_images = (np.asarray(training_images))/255
training_depths = np.asarray(training_depths)

train_images, test_images, train_depths, test_depths = train_test_split(training_images, training_depths, test_size=0.2)

train_images = train_images.reshape((len(train_images),284,75,1))
test_images = test_images.reshape((len(test_images),284,75,1))
train_depths = train_depths.reshape((len(train_depths),284,75,1))
test_depths = test_depths.reshape((len(test_depths),284,75,1))

# Normalize pixel values to be between 0 and 1
training_images = np.asarray(training_images)
training_images = training_images / 255.0
test_images = np.asarray(test_images)
test_images = test_images / 255.0
training_depths = np.asarray(training_depths)
test_labels = np.asarray(training_depths)
