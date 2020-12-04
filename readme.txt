KITTI Monocular Depth Estimation
System loads images from KITTI raw and annotated depths datasets and trains an autoencoder to produce depth values

Required data structure:
X:
data/images
place a directory for each desired scene in this folder (scenes are named similar to "2011_09_26_drive_0001_sync")

Y:
data/annotated_depths
place annotated depth maps corresponding to the scenes placed in data/images
