# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 20:20:08 2016

@author: kaavee
"""

import tensorflow as tf
from skimage import data, io, data_dir
import numpy as np

np.set_printoptions(threshold=np.nan)

print("0")
image_collect = io.imread_collection("train/*.png")
images = io.concatenate_images(image_collect)
images2=images.reshape((images.shape[0], images.shape[1]*images.shape[2]))
images2[images2==255]=1

# for i in range(320):
# 	for j in range(320):
# 		if images2[100][i*320+j]!=255:
# 			print(i,j,images2[100][i*320+j])

# print("done")
print("1")

validation_image_collect = io.imread_collection("valid/*.png")
validation_images = io.concatenate_images(validation_image_collect)
validation_images2=validation_images.reshape((validation_images.shape[0], validation_images.shape[1]*validation_images.shape[2]))
validation_images2[validation_images2==255]=1
print("2")

yinit=np.loadtxt("train/labels.txt",dtype="int") 
ys=np.zeros((yinit.shape[0],104))
ys[np.arange(yinit.shape[0]), yinit] = 1
print("3")

# for i in range(yinit.shape[0]):
# 		if ys[i][yinit[i]]==1:
# 			print(i)

# print("done")

valid_yinit=np.loadtxt("valid/labels.txt",dtype="int") 
valid_ys=np.zeros((valid_yinit.shape[0],104))
valid_ys[np.arange(valid_yinit.shape[0]), valid_yinit] = 1

print("4")

import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 102400])

W1 = tf.Variable(tf.fill([102400, 1000],value=0.1))
W1 = tf.Print(W1, [W1], message="This is W1: ", summarize = 10)
b1 = tf.Variable(tf.zeros([1000]))
b1 = tf.Print(b1, [b1], message="This is b1: ", summarize = 10)

tmp = tf.matmul(x, W1) + b1
tmp = tf.Print(tmp, [tmp], message="This is tmp: ", summarize = 10)
h1=tf.nn.relu(tmp)
h1 = tf.Print(h1, [h1], message="This is h1: ", summarize = 10)

W2 = tf.Variable(tf.fill([1000, 104],value=0.1))
W2 = tf.Print(W2, [W2], message="This is W2: ", summarize = 10)
b2 = tf.Variable(tf.zeros([104]))
b2 = tf.Print(b2, [b2], message="This is b2: ", summarize = 10)
tmpp = tf.matmul(h1, W2) + b2
tmpp = tf.Print(tmpp, [tmpp], message="This is tmpp: ", summarize = 10)
y=tf.nn.softmax(tmpp)
y = tf.Print(y, [y], message="This is y: ", summarize = 10)



y_ = tf.placeholder(tf.float32, [None, 104])
# y_ = tf.Print(y_, [y_], message="This is y_real: ", summarize = 10)

cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
cross_entropy = tf.Print(cross_entropy,[cross_entropy],"This is cross entropy: ")

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print("start")

for i in range(100):
	batch_xs = images2[(i%100)*172:((i%100)+1)*172]
	batch_ys = ys[(i%100)*172:((i%100)+1)*172]
	# print("This is y_real: ", batch_ys)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
print("finish")

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_avg(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: validation_images2, y_: valid_ys}))

sess.close()

