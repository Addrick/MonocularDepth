"""
 Loads KITTI dataset from local folders
 Expects single pool of driving images and depth pngs
 and train_test_splits them into two sets
 @author: Adam Santos
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2 as cv

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
    imgs = os.listdir(img_scene_dir)
    # remove first and last 5 images from RAW set to match annotated depth set
    del imgs[0:5]
    del imgs[-6:-1]
    depth_scene_dir = y_data_path + scene + '/proj_depth/groundtruth/image_02/'
    depths = os.listdir(depth_scene_dir)
    # iterate through files and append each example to img/depth arrays
    for img in imgs:
        impath = img_scene_dir + img
        img = cv.imread(impath, cv.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv.resize(img, dsize=(284, 75))
            images.append(img)
            images.append(img)
    for depth in depths:
        depth_maps.append(depth)

print(len(images))
print(len(depth_maps))

# load images and depth maps into arrays
# for im_name in images:
#     im = x_data_path + im_name
#     img = cv.imread(im,cv.IMREAD_GRAYSCALE)
#     if img is not None:
#         img = cv.resize(img, dsize=(284,75))
#         training_images.append(img)
# for depth in depths:
#     dm = x_data_path + depth
#     depth_map = rd.depth_read(dm)
#     if depth_map is not None:
#         depth_map = cv.resize(depth_map, dsize=(284,75))
#         training_depths.append(depth_map)
#
# input_shape = img.shape
# output_shape = depth_map.shape
# output_size = output_shape[0]*output_shape[1]
#
# training_images = (np.asarray(training_images))/255
# training_depths = np.asarray(training_depths)
#
# # Using train_test_split for early debugging; will use specific driving scenes as test set eventually
# train_images, test_images, train_depths, test_depths = train_test_split(training_images, training_depths, test_size=0.2)
#
# train_images = train_images.reshape((len(train_images),284,75,1))
# test_images = test_images.reshape((len(test_images),284,75,1))
# train_depths = train_depths.reshape((len(train_depths),284,75,1))
# test_depths = test_depths.reshape((len(test_depths),284,75,1))
#
# # Normalize pixel values to be between 0 and 1
# training_images = np.asarray(training_images)
# training_images = training_images / 255.0
# test_images = np.asarray(test_images)
# test_images = test_images / 255.0
# training_depths = np.asarray(training_depths)
# test_labels = np.asarray(training_depths)
