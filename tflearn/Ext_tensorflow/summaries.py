import tensorflow as tf
import tflearn
import numpy as np

import tflearn.datasets.mnist as mnist
# print(len(mnist.load_data(one_hot=True)))

trainX, trainY, testX, testY= mnist.load_data(one_hot=True)

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

def dnn(x):
	with tf.name_scope('Layer1'):
		W1 = tf.Variable(tf.random_normal([784,256]), name = "W1")
		b1 = tf.Variable(tf.random_normal([256]), name = "b1")
		x = tf.nn.tanh(tf.add(tf.matmul(x, W1), b1))

		tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
		tflearn.add_weights_regularizer(W1, 'L2', weight_decay = 0.001)

	with tf.name_scope('Layer2'):
		W2 = tf.Variable(tf.random_normal([256,256]), name = "W2")
		b2 = tf.Variable(tf.random_normal([256]), name = "b2")
		x = tf.nn.tanh(tf.add(tf.matmul(x,W2), b2))

		tf.add_to_collection(tf.GraphKeys.ACTIVATIONS,x)
		tflearn.add_weights_regularizer(W2, 'L2', weight_decay = 0.001)

	with tf.name_scope('Layer3'):
		W3 = tf.Variable(tf.random_normal([256,10]), name = "W3")
		b3 = tf.Variable(tf.random_normal([10]), name = "b3")

		x = tf.add(tf.matmul(x, W3), b3)

	return x

net = dnn(X)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net,Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
accuracy = tf.reduce_mean(
	tf.cast(tf.equal(tf.argmax(net,1), tf.argmax(Y,1)), tf.float32), name = "acc")

with tf.name_scope('CustomMonitor'):
	test_var = tf.reduce_sum(tf.cast(net,tf.float32), name = "test_var")
	test_const = tf.constant(32.0, name = "custom_constant")

trainop = tflearn.TrainOp(loss = loss, optimizer = optimizer, 
	metric= accuracy, batch_size = 128)
trainer = tflearn.Trainer(train_ops = trainop,
	tensorboard_dir='/tmp/tflearn_logs/', tensorboard_verbose = 2)

trainer.fit ({X: trainX, Y: trainY}, val_feed_dicts = {X: testX, Y:testY},
	n_epoch = 10, show_metric = True, run_id='Summaries_example')
