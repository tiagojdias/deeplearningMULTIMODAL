from __future__ import division, print_function, absolute_import

import tensorflow as tf
import tflearn
import numpy as np

import tflearn.datasets.mnist as mnist
# print(len(mnist.load_data(one_hot=True)))

trainX, trainY, testX, testY= mnist.load_data(one_hot=True)

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

W1 = tf.Variable(tf.random_normal([784,256]))
W2 = tf.Variable(tf.random_normal([256,256]))
W3 = tf.Variable(tf.random_normal([256,10]))
b1 = tf.Variable(tf.random_normal([256]))
b2 = tf.Variable(tf.random_normal([256]))
b3 = tf.Variable(tf.random_normal([10]))

def dnn(x):
	x = tflearn.prelu(tf.add(tf.matmul(x, W1), b1))
	tflearn.summaries.monitor_activation(x)
	x = tflearn.prelu(tf.add(tf.matmul(x, W2), b2))
	tflearn.summaries.monitor_activation(x)
	x = tf.nn.softmax(tf.add(tf.matmul(x, W3), b3))
	return x

net = dnn(X)

loss = tflearn.categorical_crossentropy(net,Y)
acc = tflearn.metrics.accuracy_op(net,Y)

optimizer = tflearn.SGD(learning_rate = 0.1, lr_decay = 0.96, decay_step = 200)

step = tflearn.variable("step", initializer ='zeros', shape = [])
optimizer.build(step_tensor=step)
optim_tensor = optimizer.get_tensor()

trainop=tflearn.TrainOp(loss = loss , optimizer = optim_tensor,
	metric = acc, batch_size = 128, step_tensor = step)

trainer = tflearn.Trainer(train_ops = trainop, tensorboard_verbose = 0)

trainer.fit ({X: trainX, Y: trainY}, val_feed_dicts = {X: testX, Y:testY},
	n_epoch = 10, show_metric = True)
