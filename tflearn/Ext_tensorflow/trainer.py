from __future__ import print_function

import tensorflow as tf
import tflearn
import numpy as np

import tflearn.datasets.mnist as mnist
# print(len(mnist.load_data(one_hot=True)))

trainX, trainY, testX, testY= mnist.load_data(one_hot=True)

X = tf.placeholder(shape=(None,784),dtype = tf.float32)
Y = tf.placeholder(shape=(None,10),dtype = tf.float32)

W1 = tf.Variable(tf.random_normal([784,256]))
W2 = tf.Variable(tf.random_normal([256,256]))
W3 = tf.Variable(tf.random_normal([256,10]))
b1 = tf.Variable(tf.random_normal([256]))
b2 = tf.Variable(tf.random_normal([256]))
b3 = tf.Variable(tf.random_normal([10]))

def dnn(x):
	x = tf.nn.tanh(tf.add(tf.matmul(x, W1), b1))
	x = tf.nn.tanh(tf.add(tf.matmul(x, W2), b2))
	x = tf.add(tf.matmul(x, W3), b3)
	return x

net = dnn(X)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net,Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
accuracy = tf.reduce_mean(
	tf.cast(tf.equal(tf.argmax(net,1), tf.argmax(Y,1)),tf.float32),name = 'acc')

trainop = tflearn.TrainOp(loss = loss, optimizer = optimizer,
	metric = accuracy, batch_size = 128)

trainer = tflearn.Trainer(train_ops = trainop, tensorboard_verbose = 3)

trainer.fit ({X: trainX, Y: trainY}, val_feed_dicts = {X: testX, Y:testY},
	n_epoch = 10, show_metric = True)
