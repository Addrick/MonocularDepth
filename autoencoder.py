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
import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2 as cv

def load_data():
    """
     Loads KITTI dataset from local folders
     Expects single pool of driving images and depth pngs
     and train_test_splits them into two sets
     @author: Adam Santos
    """
    x_data_path = "data/images/"
    y_data_path = "data/annotated_depths/"
    images = []
    depth_maps = []
    # load data from folders based on folder name:
    # organize data into subfolders by scene
    # iterate through each folder
    scenes = os.listdir(x_data_path)
    for scene in scenes:
        img_scene_dir = x_data_path + scene + '/' + scene[0:10] + '/' + scene + '/image_00/data/'
        depth_scene_dir = y_data_path + scene + '/proj_depth/groundtruth/image_02/'
        depths = os.listdir(depth_scene_dir)
        imgs = os.listdir(depth_scene_dir)

        # iterate through files and append each example to img/depth arrays
        for img in imgs:
            impath = img_scene_dir + img
            img = cv.imread(impath, cv.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv.resize(img, dsize=(284, 75))
                images.append(img)
        for depth in depths:
            impath = depth_scene_dir + depth
            depth = cv.imread(impath, cv.IMREAD_GRAYSCALE)
            if depth is not None:
                depth = cv.resize(img, dsize=(284, 75))
            depth_maps.append(depth)

    print("Num images: " + str(len(images)))
    print("Num images: " + str(len(depth_maps)))

    images = (np.asarray(images)) / 255
    depth_maps = np.asarray(depth_maps)

    # Using train_test_split for early debugging; will use specific driving scenes as test set eventually
    train_images, test_images, train_depths, test_depths = train_test_split(images, depth_maps, test_size=0.2)

    train_images = train_images.reshape((len(train_images), 75, 284, 1))
    test_images = test_images.reshape((len(test_images), 75, 284, 1))
    train_depths = train_depths.reshape((len(train_depths), 75, 284, 1))
    test_depths = test_depths.reshape((len(test_depths), 75, 284, 1))

    # Normalize pixel values to be between 0 and 1
    train_images = np.asarray(train_images)
    test_images = np.asarray(test_images)
    train_depths = np.asarray(train_depths)
    test_depths = np.asarray(test_depths)

    input_shape = img.shape
    output_shape = depth_maps.shape
    output_size = output_shape[0] * output_shape[1]

    print("Data loaded.")
    return train_images, train_depths,test_images, test_depths


def train_model(train_images, train_depths, test_images, test_depths):
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
    model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(75, 284, 1)))
    model.add(MaxPooling2D((3, 3)))
    model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D((3, 3)))
    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
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
    model.add(Reshape((75, 284, 1)))
    # Compile model
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=tf.keras.metrics.RootMeanSquaredError(name='rmse'))
    model.summary()

    history = model.fit(train_images, train_depths, batch_size=64, epochs=60,
                        validation_data=(test_images, test_depths))
    return model, history

def inference(image):
    import matplotlib.pyplot as plt
    bw_img = image.reshape(1, 75, 284, 1)
    pred = model.predict(bw_img)
    pred *= 255 / pred.max()
    pred = pred.astype(np.uint8)
    pred = pred.reshape(75,284)
    plt.imshow(image, 'gray')
    plt.imshow(pred, 'gray')
    plt.show()


train_images, train_depths, test_images, test_depths = load_data()
model, history = train_model(train_images, train_depths, test_images, test_depths)