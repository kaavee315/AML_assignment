# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 20:20:08 2016

@author: kaavee
"""

import tensorflow as tf
from skimage import data, io, data_dir
import numpy as np

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

image_collect = io.imread_collection("train/*.png")
images = io.concatenate_images(image_collect)
images2=images.reshape((images.shape[0], images.shape[1]*images.shape[2]))
images2[images2==255]=1

# for i in range(320):
# 	for j in range(320):
# 		if images2[100][i*320+j]!=255:
# 			print(i,j,images2[100][i*320+j])

# print("done")

validation_image_collect = io.imread_collection("valid/*.png")
validation_images = io.concatenate_images(validation_image_collect)
validation_images2=validation_images.reshape((validation_images.shape[0], validation_images.shape[1]*validation_images.shape[2]))
validation_images2[validation_images2==255]=1

yinit=np.loadtxt("train/labels.txt",dtype="int") 
ys=np.zeros((yinit.shape[0],104))
ys[np.arange(yinit.shape[0]), yinit] = 1

# for i in range(yinit.shape[0]):
# 		if ys[i][yinit[i]]==1:
# 			print(i)

# print("done")

valid_yinit=np.loadtxt("valid/labels.txt",dtype="int") 
valid_ys=np.zeros((valid_yinit.shape[0],104))
valid_ys[np.arange(valid_yinit.shape[0]), valid_yinit] = 1

import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 102400])

W = tf.Variable(tf.fill([102400, 104],value=0.1),
                      name="weights")
# W = tf.Print(W, [W], message="This is W: ")
b = tf.Variable(tf.zeros([104]))
# b = tf.Print(b, [b], message="This is b: ")

tmp = tf.matmul(x, W) + b
# tmp = tf.Print(tmp, [tmp], message="This is tmp: ")
total=tf.reduce_sum(tmp)
y = tf.nn.softmax(tmp)
# y = tf.Print(y, [y], message="This is y: ")


y_ = tf.placeholder(tf.float32, [None, 104])

cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
# cross_entropy = tf.Print(cross_entropy,[cross_entropy],"This is cross entropy:")

train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print("start")

for i in range(100):
	batch_xs = images2[(i%100)*172:((i%100)+1)*172]
	batch_ys = ys[(i%100)*172:((i%100)+1)*172]
	# print(batch_ys)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: validation_images2, y_: valid_ys}))

sess.close()

