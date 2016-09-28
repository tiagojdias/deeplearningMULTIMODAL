import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time
import numpy as np
import scipy.misc as misc
import scipy.ndimage as ndimage
from scipy.signal import convolve2d
import pylab as plt
import random
import math
import scipy.io as sio

# Image Definitions
image_size = 56
num_labels = 3
imgSize = 56
imgSize_flat = imgSize * imgSize
imgShape = (imgSize, imgSize)
num_channels = 1  # Gray-scale
num_classes = 3
priorClasses = np.array([0.5,0.25,0.25]);

# Training + validation and Test sizes
nmbTrainImg = 1023
nmbValImg = 723
nmbTestImg = 501


mat_contents = sio.loadmat('Trn_stuff.mat')
# print(mat_contents.keys())
trainImg = mat_contents['trainImg']
trainMask = mat_contents['trainMask']
trainClass = mat_contents['trainClass']
# print(trainImg.shape)
# print(trainMask.shape)
# print(trainClass.shape)

mat_contents = sio.loadmat('Val_stuff.mat')
# print(mat_contents.keys())
validImg = mat_contents['valImg']
validMask = mat_contents['valMask']
validClass = mat_contents['valClass']
# print(validImg.shape)
# print(validMask.shape)
# print(validClass.shape)

mat_contents = sio.loadmat('Tst_stuff.mat')
# print(mat_contents.keys())
testImg = mat_contents['testImg']
testMask = mat_contents['testMask']
testClass = mat_contents['testClass']

trainClass[0,:]=trainClass[0,:]-1
validClass[0,:]=validClass[0,:]-1
testClass[0,:]=testClass[0,:]-1

# print (trainImg.shape)
CASE = 1

if CASE == 1:
	train_dataset = trainImg.reshape(
		(-1,imgSize,imgSize,num_channels)).astype(np.float32)
	valid_dataset = validImg.reshape(
		(-1,imgSize,imgSize,num_channels)).astype(np.float32)
	test_dataset = testImg.reshape(
		(-1,imgSize,imgSize,num_channels)).astype(np.float32)
else:
	train_dataset = trainMask.reshape(
		(-1,imgSize,imgSize,num_channels)).astype(np.float32)
	valid_dataset = validMask.reshape(
		(-1,imgSize,imgSize,num_channels)).astype(np.float32)
	test_dataset = testMask.reshape(
		(-1,imgSize,imgSize,num_channels)).astype(np.float32)


trainClass = np.squeeze(np.asarray(trainClass))
validClass = np.squeeze(np.asarray(validClass))
testClass = np.squeeze(np.asarray(testClass))
train_labels = (np.arange(num_classes) == trainClass[:,None]).astype(np.float32)
valid_labels = (np.arange(num_classes) == validClass[:,None]).astype(np.float32)
test_labels = (np.arange(num_classes) == testClass[:,None]).astype(np.float32)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


x = tf.placeholder(tf.float32,  shape=[None, imgSize, imgSize, num_channels])
y_ = tf.placeholder(tf.float32, shape=[None, 3])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev = 0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape = shape)
  return tf.Variable(initial)

def conv2d (x, W):
  return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = "VALID")

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize= [1,2,2,1], strides = [1,2,2,1], padding ="SAME" )

W_conv1 = weight_variable([5,5,1,20])
b_conv1 = bias_variable([20])

# x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = conv2d(x,W_conv1) + b_conv1
print(h_conv1)
h_pool1 = max_pool_2x2(h_conv1)
print(h_pool1)

W_conv2 = weight_variable([7,7,20,50])
b_conv2 = bias_variable([50])

h_conv2 = conv2d(h_pool1,W_conv2) + b_conv2
print(h_conv2)
h_pool2 = max_pool_2x2(h_conv2)
print(h_pool2)

W_conv3 = weight_variable([10,10,50,500])
b_conv3 = bias_variable([500])

h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3) + b_conv3)
print(h_conv3)

h_conv3_flat = tf.reshape(h_conv3,[-1, 1*1*500])
print(h_conv3_flat)

W_fc1 = weight_variable([500,3])
b_fc1 = bias_variable([3])

y_conv = tf.nn.softmax(tf.matmul(h_conv3_flat,W_fc1) + b_fc1)
print(y_conv)



# w_fc1 = weight_variable([7*7*64,1024])
# b_fc1 = bias_variable([1024])

# h_pool2_flat = tf.reshape(h_pool2,[-1, 7*7*64])
# print(h_pool2_flat)
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1) + b_fc1)
# print(h_fc1)

keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)


# W_fc2 = weight_variable([1024,10])
# b_fc2 = bias_variable([10])

# y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)
# print(y_conv)

batch_size = 100

cross_entropy = tf.reduce_mean(-tf.reduce_sum(
  y_ * tf.log(y_conv), reduction_indices = [1]))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  for i in range(200):
    # batch = mnist.train.next_batch(50)
    offset = (i * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    

    if i%10 == 0:
      train_accuracy = accuracy.eval(
        feed_dict={x:batch_data, y_:batch_labels, keep_prob: 0.0})
      print ("step %d, training accuracy %g"%(i,train_accuracy))
    train_step.run(feed_dict={x: batch_data, y_:batch_labels, keep_prob: 0.0})

  print ("test accuracy %g" %accuracy.eval(
    feed_dict={x:test_dataset, y_: test_labels, keep_prob: 0.0}))
