# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 20:20:08 2016

@author: kaavee
"""

import tensorflow as tf
from skimage import data, io, data_dir, transform
import numpy as np
<<<<<<< HEAD
import random

def flatten2(images):
	images2 = np.zeros((images.shape[0],images.shape[1]*images.shape[2]))
	for i in range(images.shape[0]):
		images2[i]=images[i].flatten()
	return images2

=======
>>>>>>> 8aca118790cfe88eca36160102ad9ff010555eae

np.set_printoptions(threshold=np.nan)

print("0")

image_collect = io.imread_collection("train/*.png")
images = io.concatenate_images(image_collect)
images1 = np.zeros((images.shape[0],50,50))
for i in range(images.shape[0]):
	images1[i] = transform.resize(images[i],[50,50])
<<<<<<< HEAD
images2=flatten2(images)
=======
images2=images1.reshape((images1.shape[0], images1.shape[1]*images1.shape[2]))

# for i in range(320):
# 	for j in range(320):
# 		if images2[100][i*320+j]!=255:
# 			print(i,j,images2[100][i*320+j])

# print("done")
print("1")
>>>>>>> 8aca118790cfe88eca36160102ad9ff010555eae

validation_image_collect = io.imread_collection("valid/*.png")
validation_images = io.concatenate_images(validation_image_collect)
validation_images1 = np.zeros((validation_images.shape[0],50,50))
for i in range(validation_images.shape[0]):
	validation_images1[i] = transform.resize(validation_images[i],[50,50])
<<<<<<< HEAD
validation_images2=flatten2(validation_images)	
=======
validation_images2=validation_images1.reshape((validation_images1.shape[0], validation_images1.shape[1]*validation_images1.shape[2]))
print("2")
>>>>>>> 8aca118790cfe88eca36160102ad9ff010555eae

yinit=np.loadtxt("train/labels.txt",dtype="int") 
ys=np.zeros((yinit.shape[0],104))
ys[np.arange(yinit.shape[0]), yinit] = 1
print("3")

<<<<<<< HEAD
=======
# for i in range(yinit.shape[0]):
# 		if ys[i][yinit[i]]==1:
# 			print(i)

# print("done")

>>>>>>> 8aca118790cfe88eca36160102ad9ff010555eae
valid_yinit=np.loadtxt("valid/labels.txt",dtype="int") 
valid_ys=np.zeros((valid_yinit.shape[0],104))
valid_ys[np.arange(valid_yinit.shape[0]), valid_yinit] = 1

print("4")

<<<<<<< HEAD
# import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 2500])


W0 = tf.Variable(tf.fill([2500, 1000],value=0.001))
# W0 = tf.Print(W0, [W0], message="This is W0: ", summarize = 10)
b0 = tf.Variable(tf.fill([1000],value=0.001))
# b0 = tf.Print(b0, [b0], message="This is b0: ", summarize = 10)
z0 = tf.matmul(x, W0) + b0
# z0 = tf.Print(z0, [z0], message="This is z0: ", summarize = 10)
h0=tf.nn.relu(z0)
# h0 = tf.Print(h0, [h0], message="This is h0: ", summarize = 10)


W1 = tf.Variable(tf.fill([1000, 500],value=0.001))
# W1 = tf.Print(W1, [W1], message="This is W1: ", summarize = 10)
b1 = tf.Variable(tf.fill([500],value=0.001))
# b1 = tf.Print(b1, [b1], message="This is b1: ", summarize = 10)
z1 = tf.matmul(h0, W1) + b1
# z1 = tf.Print(z1, [z1], message="This is z1: ", summarize = 10)
h1=tf.nn.relu(z1)
# h1 = tf.Print(h1, [h1], message="This is h1: ", summarize = 10)

W2 = tf.Variable(tf.fill([500, 104],value=0.001))
# W2 = tf.Print(W2, [W2], message="This is W2: ", summarize = 10)
b2 = tf.Variable(tf.fill([104],value=0.001))
# b2 = tf.Print(b2, [b2], message="This is b2: ", summarize = 10)
z2 = tf.matmul(h1, W2) + b2
# z2 = tf.Print(z2, [z2], message="This is z2: ", summarize = 10)
y=tf.nn.softmax(z2)
# y_reduce = tf.reduce_sum(y,1)
# y_reduce = tf.Print(y_reduce, [y_reduce], message="This is y_reduce: ", summarize = 100000)
=======
import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 2500])

W1 = tf.Variable(tf.fill([2500, 500],value=0.1))
# W1 = tf.Print(W1, [W1], message="This is W1: ", summarize = 10)
b1 = tf.Variable(tf.zeros([500]))
# b1 = tf.Print(b1, [b1], message="This is b1: ", summarize = 10)

tmp = tf.matmul(x, W1) + b1
# tmp = tf.Print(tmp, [tmp], message="This is tmp: ", summarize = 10)
h1=tf.nn.relu(tmp)
# h1 = tf.Print(h1, [h1], message="This is h1: ", summarize = 10)

W2 = tf.Variable(tf.fill([500, 104],value=0.1))
# W2 = tf.Print(W2, [W2], message="This is W2: ", summarize = 10)
b2 = tf.Variable(tf.zeros([104]))
# b2 = tf.Print(b2, [b2], message="This is b2: ", summarize = 10)
tmpp = tf.matmul(h1, W2) + b2
# tmpp = tf.Print(tmpp, [tmpp], message="This is tmpp: ", summarize = 10)
y=tf.nn.softmax(tmpp)
# y = tf.Print(y, [y], message="This is y: ", summarize = 10)
>>>>>>> 8aca118790cfe88eca36160102ad9ff010555eae



y_ = tf.placeholder(tf.float32, [None, 104])
# y_ = tf.Print(y_, [y_], message="This is y_real: ", summarize = 10)

cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
# cross_entropy = tf.Print(cross_entropy,[cross_entropy],"This is cross entropy: ")

<<<<<<< HEAD
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
=======
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
>>>>>>> 8aca118790cfe88eca36160102ad9ff010555eae

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print("start")

for i in range(100):
<<<<<<< HEAD

	batch_xs = np.zeros((100,2500))
	batch_ys =np.zeros((100,104))
	for j in range(100):
		a=random.randrange(0,17204,1)
		if(i==0):
			print(a)
		batch_ys[i]=ys[a]
=======
	batch_xs = images2[(i%100)*172:((i%100)+1)*172]
	batch_ys = ys[(i%100)*172:((i%100)+1)*172]
>>>>>>> 8aca118790cfe88eca36160102ad9ff010555eae
	# print("This is y_real: ", batch_ys)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
print("finish")

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: validation_images2, y_: valid_ys}))

sess.close()

