"""
 Autoencoder for Monocular Depth Estimation
 @author: Adam Santos

 IDEAS:
 - depth prediction looks very low resolution
 - current network is too small for the task? (but not enough GPU memory to run anything larger)
 - problems from downsampling the data by 5? (not enough memory to run them at higher resolution)
 - problems in data encoding/reading? depth png's are 16-bit I just ignore that
"""

from keras.constraints import maxnorm
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Reshape, UpSampling2D, Cropping2D, \
    AveragePooling2D
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import load_model
import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2 as cv
import matplotlib.pyplot as plt
import plot_history as ph


def load_data():
    """
     Loads KITTI dataset from local folders
    """
    x_data_path = "data/images/"
    y_data_path = "data/annotated_depths/"
    images = []
    depth_maps = []
    # iterate through each scene folder in data_paths
    scenes = os.listdir(x_data_path)
    for scene in scenes:
        img_scene_dir = x_data_path + scene + '/' + scene[0:10] + '/' + scene + '/image_00/data/'
        depth_scene_dir = y_data_path + scene + '/proj_depth/groundtruth/image_02/'
        depths = os.listdir(depth_scene_dir)
        imgs = os.listdir(depth_scene_dir)

        # iterate through files and append each example to img/depth arrays
        for img in depths:
            impath = img_scene_dir + img
            img = cv.imread(impath, cv.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv.resize(img, dsize=(284, 75))
                images.append(np.asarray(img))
        for depth in depths:
            impath = depth_scene_dir + depth
            depth = cv.imread(impath, cv.IMREAD_GRAYSCALE)
            if depth is not None:
                depth = cv.resize(depth, dsize=(284, 75))
                depth_maps.append(np.asarray(depth))

    # Normalize image pixel values
    # images = (np.asarray(images)) / 255
    images = (np.asarray(images))
    depth_maps = np.asarray(depth_maps)

    # Using train_test_split for early debugging; will use specific driving scenes as test set eventually
    train_images, test_images, train_depths, test_depths = train_test_split(images, depth_maps, test_size=0.2)

    # Increase ndim to 4 for network
    dim = train_images.shape
    train_images = train_images.reshape((len(train_images), dim[1], dim[2], 1))
    test_images = test_images.reshape((len(test_images), dim[1], dim[2], 1))
    train_depths = train_depths.reshape((len(train_depths), dim[1], dim[2], 1))
    test_depths = test_depths.reshape((len(test_depths), dim[1], dim[2], 1))

    # Normalize pixel values to be between 0 and 1
    train_images = np.asarray(train_images)
    test_images = np.asarray(test_images)
    train_depths = np.asarray(train_depths)
    test_depths = np.asarray(test_depths)

    print("Data loaded.")
    return train_images, train_depths,test_images, test_depths

def train_model(train_images, train_depths, test_images, test_depths, save_best=True):
    # Initializes GPU to allow memory growth instead of static allocation
    # Resolves some intermittent initialization errors
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    print("Training...")
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # Create the model
    # Convolutional autoencoder to transform a grayscale image to an inferred depth map
    model = Sequential()
    model.add(Conv2D(64, kernel_size=7, padding='same', activation='relu', input_shape=(75, 284, 1)))
    model.add(MaxPooling2D((3, 3), padding='same'))
    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    # model.add(MaxPooling2D((2, 2), padding='same'))
    # model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    # model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))

    model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
    # model.add(UpSampling2D((2, 2)))
    # model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
    # model.add(UpSampling2D((2, 2)))
    # model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
    model.add(UpSampling2D((3, 3)))
    model.add(Conv2D(1, kernel_size=1, activation='relu', padding='same'))
    model.add(Cropping2D(cropping=((0, 0), (0, 1)), data_format=None))

    # Compile model
    model.compile(optimizer='adadelta',
                  loss='mse',
                  metrics=tf.keras.metrics.RootMeanSquaredError(name='rmse'))

    # Save the weights with lowest RMSE
    callbacks_list = []
    if save_best:
        filepath = "best_depth_weights_last_run.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_rmse', verbose=1, save_best_only=True, mode='min')
        callbacks_list.append(checkpoint)

    # Train model
    history = model.fit(train_images, train_depths, batch_size=128, epochs=1500,
                        validation_data=(test_images, test_depths), verbose=1, callbacks=callbacks_list)
    return model, history

def sample_inferences():
    # Runs 3 predictions on the model and plots image, actual depth, and prediction for each
    bw_img1 = test_images[0].reshape(1, 75, 284, 1)
    bw_img2 = test_images[10].reshape(1, 75, 284, 1)
    bw_img3 = test_images[20].reshape(1, 75, 284, 1)

    pred = model.predict(bw_img1)
    pred *= 255 / pred.max()
    pred = pred.astype(np.uint8)
    pred = pred.reshape(75,284)

    pred2 = model.predict(bw_img2)
    pred2 *= 255 / pred2.max()
    pred2 = pred2.astype(np.uint8)
    pred2 = pred2.reshape(75,284)

    pred3 = model.predict(bw_img3)
    pred3 *= 255 / pred3.max()
    pred3 = pred3.astype(np.uint8)
    pred3 = pred3.reshape(75,284)

    plt.figure(figsize=(16, 6))
    ax = plt.subplot(3,3,1)
    ax.set_title("Image Input 1")
    plt.imshow(test_images[0], 'gray')
    ax = plt.subplot(3,3,2)
    ax.set_title("Ground Truth Depth 1")
    plt.imshow(test_depths[0], 'gray')
    ax = plt.subplot(3,3,3)
    ax.set_title("Predicted Depth 1")
    plt.imshow(pred, 'gray')

    ax = plt.subplot(3,3,4)
    ax.set_title("Image Input 2")
    plt.imshow(test_images[10], 'gray')
    ax = plt.subplot(3,3,5)
    ax.set_title("Ground Truth Depth 2")
    plt.imshow(test_depths[10], 'gray')
    ax = plt.subplot(3,3,6)
    ax.set_title("Predicted Depth 2")
    plt.imshow(pred2, 'gray')

    ax = plt.subplot(3,3,7)
    ax.set_title("Image Input 3")
    plt.imshow(test_images[20], 'gray')
    ax = plt.subplot(3,3,8)
    ax.set_title("Ground Truth Depth 3")
    plt.imshow(test_depths[20], 'gray')
    ax = plt.subplot(3,3,9)
    ax.set_title("Predicted Depth 3")
    plt.imshow(pred3, 'gray')
    plt.show()

def check_data(train_images, train_depths, test_images, test_depths):
    # Plot image/depth pair
    plt.figure(figsize=(16,6))
    ax = plt.subplot(2,2,1)
    ax.set_title("Train Image")
    plt.imshow(train_images[10], 'gray')
    ax = plt.subplot(2,2,2)
    ax.set_title("Train Depth")
    plt.imshow(train_depths[10], 'gray')
    ax = plt.subplot(2,2,3)
    ax.set_title("Test Image")
    plt.imshow(test_images[10], 'gray')
    ax = plt.subplot(2,2,4)
    ax.set_title("Test Depth")
    plt.imshow(test_depths[10], 'gray')
    plt.show()

def load_weights():
    # load weights into new model
    loaded_model = load_model("depth_48776_val_rmse.hdf5")
    print("Loaded model from disk")
    # Compile model
    loaded_model.compile(optimizer='adadelta',
                  loss='mse',
                  metrics=tf.keras.metrics.RootMeanSquaredError(name='rmse'))
    return loaded_model


try:
    model, history = train_model(train_images, train_depths, test_images, test_depths)
    sample_inferences()
except NameError:
    train_images, train_depths, test_images, test_depths = load_data()
    check_data(train_images, train_depths, test_images, test_depths)
    model, history = train_model(train_images, train_depths, test_images, test_depths)
    sample_inferences()
    ph.plot_loss(history)
    ph.plot_rmse(history)