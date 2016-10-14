import tensorflow as tf
from skimage import data, io, data_dir, transform, viewer, morphology
import numpy as np
import random
import sys
import os
import scipy
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import toimage

np.set_printoptions(threshold=np.nan)


def flatten2(images):
	images2 = np.zeros((images.shape[0],images.shape[1]*images.shape[2]))
	for i in range(images.shape[0]):
		images2[i]=images[i].flatten()
	return images2

def onehot(labels, values):
	labels2 = np.zeros((labels.shape[0], values))
	for i in range(labels.shape[0]):
		labels2[i][labels[i]]=1
	return labels2



print("0")

image_collect = io.imread_collection("train/*.png")
# view_coll = viewer.CollectionViewer(image_collect);
# view_coll.show()
images = io.concatenate_images(image_collect)
# v=viewer.ImageViewer(images[1])
# v.show()

final_dim=64

# images1 = np.zeros((images.shape[0],final_dim,final_dim))
# for i in range(images.shape[0]):
# 	tmp_image=np.invert(images[i])
# 	tmp_image=morphology.dilation(tmp_image)
# 	tmp_image=np.invert(tmp_image)
# 	images1[i] = transform.resize(tmp_image,[final_dim,final_dim])
# v2=viewer.ImageViewer(images1[0])
# v2.show()
# images2=flatten2(images1)

images1 = np.zeros((images.shape[0],final_dim,final_dim))
for i in range(images.shape[0]):
	tmp_image = gaussian_filter(images[i], sigma=3)
	tmp_image = 255 - tmp_image
	tmp_image = scipy.misc.imresize(arr=tmp_image, size=(final_dim, final_dim))
	images1[i] = tmp_image
	if(i==0):
		v2=viewer.ImageViewer(images1[0])
		v2.show()
images2=flatten2(images1)


validation_image_collect = io.imread_collection("valid/*.png")
validation_images = io.concatenate_images(validation_image_collect)
# v=viewer.ImageViewer(validation_images[validation_images.shape[0]-1])
# v.show()
validation_images1 = np.zeros((validation_images.shape[0],final_dim,final_dim))
for i in range(validation_images.shape[0]):
	tmp_image = gaussian_filter(validation_images[i], sigma=3)
	tmp_image = 255 - tmp_image
	tmp_image = scipy.misc.imresize(arr=tmp_image, size=(final_dim, final_dim))
	validation_images1[i] = tmp_image
# v2=viewer.ImageViewer(validation_images1[validation_images1.shape[0]-1])
# v2.show()
validation_images2=flatten2(validation_images1)

np.save('train_preprocessed', images2)
np.save('valid_preprocessed', validation_images2)