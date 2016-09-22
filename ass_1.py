# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 20:20:08 2016

@author: kaavee
"""

import tensorflow as tf
from skimage import data, io, data_dir
import numpy as np

image_collect = io.imread_collection("train/*.png")
images = io.concatenate_images(image_collect)
images2=images.reshape((images.shape[0], images.shape[1]*images.shape[2]))

validation_image_collect = io.imread_collection("valid/*.png")
validation_images = io.concatenate_images(validation_image_collect)
validation_images2=validation_images.reshape((validation_images.shape[0], validation_images.shape[1]*validation_images.shape[2]))

yinit=np.loadtxt("train/labels.txt",dtype="int") 
ys=np.zeros((yinit.shape[0],104))
ys[np.arange(yinit.shape[0]), yinit] = 1

valid_yinit=np.loadtxt("valid/labels.txt",dtype="int") 
valid_ys=np.zeros((valid_yinit.shape[0],104))
valid_ys[np.arange(valid_yinit.shape[0]), valid_yinit] = 1

import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 102400])

W = tf.Variable(tf.zeros([102400, 104]))
b = tf.Variable(tf.zeros([104]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 104])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(100):
	batch_xs = images2[i*172:(i+1)*172]
	batch_ys = ys[i*172:(i+1)*172]
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: validation_images2, y_: valid_ys}))

sess.close()

