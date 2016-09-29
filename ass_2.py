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
images2[images2>125]=1

validation_image_collect = io.imread_collection("valid/*.png")
validation_images = io.concatenate_images(validation_image_collect)
validation_images2=validation_images.reshape((validation_images.shape[0], validation_images.shape[1]*validation_images.shape[2]))
validation_images2[validation_images2>125]=1

yinit=np.loadtxt("train/labels.txt",dtype="int") 

# for i in range(yinit.shape[0]):
# 		if ys[i][yinit[i]]==1:
# 			print(i)

# print("done")

valid_yinit=np.loadtxt("valid/labels.txt",dtype="int") 

x = tf.placeholder(tf.float32, [None, 102400])

W1 = tf.Variable(tf.fill([102400,16],value=0.1))
W1 = tf.Print(W1, [W1], message="This is W1: ")
b1 = tf.Variable(tf.fill([16],value=0.1))
b1 = tf.Print(b1, [b1], message="This is b1: ")
z1 = tf.matmul(x, W1) + b1
z1 = tf.Print(z1, [z1], message="This is z1: ")
h1 = tf.nn.relu(z1)
h1 = tf.Print(h1, [h1], message="This is h1: ")

W2 = tf.Variable(tf.fill([16,1],value=0.1))
W2 = tf.Print(W2, [W2], message="This is W2: ")
b2 = tf.Variable(tf.fill([1],value=0.1))
b2 = tf.Print(b2, [b2], message="This is b2: ")
y = tf.matmul(h1, W2) + b2
y = tf.Print(y, [y], message="This is y: ")

y_ = tf.placeholder(tf.float32, [None])

cross_entropy = tf.reduce_sum((y-y_)*(y-y_))

# print("hi")
print(tf.trainable_variables())
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print("start")

for i in range(5):
	batch_xs = images2[(i%100)*172:((i%100)+1)*172]
	batch_ys = yinit[(i%100)*172:((i%100)+1)*172]
	# print(batch_ys)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

print("finish")


correct_prediction = tf.reduce_sum((y-y_)*(y-y_))


print(sess.run(correct_prediction, feed_dict={x: validation_images2, y_: valid_yinit}))

sess.close()

