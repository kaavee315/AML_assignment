# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 20:20:08 2016
@author: kaavee
"""

import tensorflow as tf
from skimage import data, io, data_dir, transform, viewer, morphology
import numpy as np
import random
import sys

np.set_printoptions(threshold=np.nan)


# def flatten2(images):
# 	images2 = np.zeros((images.shape[0],images.shape[1]*images.shape[2]))
# 	for i in range(images.shape[0]):
# 		images2[i]=images[i].flatten()
# 	return images2

def onehot(labels, values):
	labels2 = np.zeros((labels.shape[0], values))
	for i in range(labels.shape[0]):
		labels2[i][labels[i]]=1
	return labels2



print("0")

# image_collect = io.imread_collection("train/*.png")
# # view_coll = viewer.CollectionViewer(image_collect);
# # view_coll.show()
# images = io.concatenate_images(image_collect)
# # v=viewer.ImageViewer(images[1])
# # v.show()

final_dim=64

# images1 = np.zeros((images.shape[0],final_dim,final_dim))
# for i in range(images.shape[0]):
# 	tmp_image=np.invert(images[i])
# 	tmp_image=morphology.dilation(tmp_image)
# 	tmp_image=np.invert(tmp_image)
# 	images1[i] = transform.resize(tmp_image,[final_dim,final_dim])
# # v2=viewer.ImageViewer(images1[images1.shape[0]-1])
# # v2.show()
# images2=flatten2(images1)


# validation_image_collect = io.imread_collection("valid/*.png")
# validation_images = io.concatenate_images(validation_image_collect)
# # v=viewer.ImageViewer(validation_images[validation_images.shape[0]-1])
# # v.show()
# validation_images1 = np.zeros((validation_images.shape[0],final_dim,final_dim))
# for i in range(validation_images.shape[0]):
# 	tmp_image=np.invert(validation_images[i])
# 	tmp_image=morphology.dilation(tmp_image)
# 	tmp_image=np.invert(tmp_image)
# 	validation_images1[i] = transform.resize(tmp_image,[final_dim,final_dim])
# # v2=viewer.ImageViewer(validation_images1[validation_images1.shape[0]-1])
# # v2.show()
# validation_images2=flatten2(validation_images1)

images2 = np.load('train_preprocessed.npy')
validation_images2 = np.load('valid_preprocessed.npy')

yinit=np.loadtxt("train/labels.txt")
ys=onehot(yinit,104)

valid_yinit=np.loadtxt("valid/labels.txt")
valid_ys=onehot(valid_yinit,104) 

print("1")

x = tf.placeholder(tf.float32, [None, final_dim*final_dim])

hl_1=int(sys.argv[1])
hl_2=int(sys.argv[2])
learning_rate = float(sys.argv[3])
out_l=104

W0 = tf.Variable(tf.random_normal([final_dim*final_dim, hl_1],mean=0.00, stddev=0.0001))
# W0 = tf.Print(W0, [W0], message="This is W0: ", summarize = 10)
b0 = tf.Variable(tf.random_normal([hl_1],mean=0.00, stddev=0.0001))
# b0 = tf.Print(b0, [b0], message="This is b0: ", summarize = 10)
z0 = tf.matmul(x, W0) + b0
# z0 = tf.Print(z0, [z0], message="This is z0: ", summarize = 10)
h0=tf.nn.relu(z0)
# h0 = tf.Print(h0, [h0], message="This is h0: ", summarize = 104)


W1 = tf.Variable(tf.random_normal([hl_1, hl_2],mean=0.00, stddev=0.0001))
# W1 = tf.Print(W1, [W1], message="This is W1: ", summarize = 10)
b1 = tf.Variable(tf.random_normal([hl_2],mean=0.00, stddev=0.0001))
# b1 = tf.Print(b1, [b1], message="This is b1: ", summarize = 10)
z1 = tf.matmul(h0, W1) + b1
# z1 = tf.Print(z1, [z1], message="This is z1: ", summarize = 10)
h1=tf.nn.relu(z1)
# y = tf.Print(y, [y], message="This is y: ", summarize = 10)

W2 = tf.Variable(tf.random_normal([hl_2, out_l],mean=0.00, stddev=0.0001))
# W2 = tf.Print(W2, [W2], message="This is W2: ", summarize = 10)
b2 = tf.Variable(tf.random_normal([out_l],mean=0.00, stddev=0.0001))
# b2 = tf.Print(b2, [b2], message="This is b2: ", summarize = 10)
z2 = tf.matmul(h1, W2) + b2
# z2 = tf.Print(z2, [z2], message="This is z2: ", summarize = 10)
y=z2
# y_reduce = tf.reduce_sum(y,1)
# y_reduce = tf.Print(y_reduce, [y_reduce], message="This is y_reduce: ", summarize = 100000)



y_ = tf.placeholder(tf.float32, [None, out_l])
# y_ = tf.Print(y_, [y_], message="This is y_real: ", summarize = 10)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
# cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
# cross_entropy = tf.Print(cross_entropy,[cross_entropy],"This is cross entropy: ")

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print("start")

iterations1=5000
iterations=100

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(iterations1):
	if((i%100)==0):
		print(i)
	sample_size=1200	
	batch_xs = np.zeros((sample_size,final_dim*final_dim))
	batch_ys =np.zeros((sample_size,out_l))
	for j in range(sample_size):
		a=random.randrange(0,17204,1)
		# if(i==0):
		# 	print(a)
		batch_xs[j]=images2[a]
		batch_ys[j]=ys[a]
	# print("This is y_real: ", batch_ys)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
	if((i%100)==0):
		print(sess.run(accuracy, feed_dict={x: validation_images2, y_: valid_ys}))

# for i in range(iterations):
# 	sample_size=1000	
# 	no_sample = int(len(images2)/sample_size)
# 	batch_xs = np.zeros((sample_size,final_dim*final_dim))
# 	batch_ys =np.zeros((sample_size,out_l))
# 	avg_crossentropy = 0
# 	for j in range(no_sample):
# 		# if(i==0):
# 		# 	print(a)
# 		batch_xs=images2[j*sample_size : (j+1)*sample_size]
# 		batch_ys=ys[j*sample_size : (j+1)*sample_size]
# 	# print("This is y_real: ", batch_ys)
# 		_,c = sess.run([train_step,cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
# 		avg_crossentropy += c/no_sample


	# if((i%10)==0):
	# 	print(i,":")
	# 	print(avg_crossentropy)
	# 	print(sess.run(accuracy, feed_dict={x: validation_images2, y_: valid_ys}))
print("finish")

print(sess.run(accuracy, feed_dict={x: validation_images2, y_: valid_ys}))

sess.close()