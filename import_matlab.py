import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time
import numpy as np
import scipy.misc as misc
import scipy.ndimage as ndimage
from scipy.signal import convolve2d
import random
import math
import scipy.io as sio


# Image Definition
imgSize = 56
imgSize_flat = imgSize * imgSize
imgShape = (imgSize, imgSize)
num_channels = 1  # Gray-scale
num_classes = 3
priorClasses = np.array([0.5, 0.25, 0.25]);

# Training + validation and Test sizes
nmbTrainImg = 1023
nmbValImg = 723
nmbTestImg = 501


mat_contents = sio.loadmat('Trn_stuff.mat')
# print(mat_contents.keys())
trainImgAux = mat_contents['trainImg']
trainMaskAux = mat_contents['trainMask']
trainClass = mat_contents['trainClass']


mat_contents = sio.loadmat('Val_stuff.mat')
# print(mat_contents.keys())
validImgAux = mat_contents['valImg']
validMaskAux = mat_contents['valMask']
validClass = mat_contents['valClass']

mat_contents = sio.loadmat('Tst_stuff.mat')
# print(mat_contents.keys())
testImgAux = mat_contents['testImg']
testMaskAux = mat_contents['testMask']
testClass = mat_contents['testClass']

#Change Matlal classes in range(0,2) from range(1,3)
trainClass = trainClass - 1
validClass = validClass - 1
testClass = testClass - 1

trainImg = np.zeros(
	[trainImgAux.shape[2], imgSize, imgSize, num_channels], dtype='float32')
trainMask = np.zeros(
	[trainMaskAux.shape[2], imgSize, imgSize, num_channels], dtype="float32")

validImg = np.zeros(
	[validImgAux.shape[2], imgSize, imgSize, num_channels], dtype='float32')
validMask = np.zeros(
	[validMaskAux.shape[2], imgSize, imgSize, num_channels], dtype="float32")

testImg = np.zeros(
	[testImgAux.shape[2], imgSize, imgSize, num_channels], dtype='float32')
testMask = np.zeros(
	[testMaskAux.shape[2], imgSize, imgSize, num_channels], dtype="float32")

# print(trainImg.shape, trainMask.shape)
# print(validImg.shape, validMask.shape)
# print(testImg.shape, testMask.shape)

for idx1 in range(nmbTrainImg):
	trainImg[idx1, :, :, 0] = trainImgAux[:, :, idx1]
	trainMask[idx1, :, :, 0] = trainMaskAux[:, :, idx1]

for idx2 in range(nmbValImg):
	validImg[idx2, :, :, 0] = validImgAux[:, :, idx2]
	validMask[idx2, :, :, 0] = validMaskAux[:, :, idx2]

for idx3 in range(nmbTestImg):
	testImg[idx3, :, :, 0] = testImgAux[:, :, idx3]
	testMask[idx3, :, :, 0] = testMaskAux[:, :, idx3]

trainClass = np.squeeze(np.asarray(trainClass))
validClass = np.squeeze(np.asarray(validClass))
testClass = np.squeeze(np.asarray(testClass))

trainClass = (np.arange(num_classes) == trainClass[:, None]).astype(np.float32)
validClass = (np.arange(num_classes) == validClass[:, None]).astype(np.float32)
testClass = (np.arange(num_classes) == testClass[:, None]).astype(np.float32)

# #######################################################################
# TensorFlow Graph
def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.scalar_summary('stddev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

def convolution_layer(input, num_input_channels, filter_size, \
	num_filters, use_pooling, use_relu, is_train):
	shape = [filter_size, filter_size, num_input_channels, num_filters]

	if is_train == True:
		with tf.variable_scope("w_and_b"):
			weights = tf.get_variable(
				"weights", shape, \
				initializer=tf.random_normal_initializer(0, 0.01))
		    # Create variable named "biases".
			biases = tf.get_variable(
			    "biases", [num_filters], \
			    initializer=tf.constant_initializer(0.0))
	else:
		with tf.variable_scope("w_and_b", reuse = True):
			weights = tf.get_variable("weights")
			biases = tf.get_variable("biases")

	layer = tf.nn.conv2d(input, weights, [1, 1, 1, 1], 'VALID') + biases

	if use_pooling:
		layer = tf.nn.max_pool(layer, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

	if use_relu:
		layer = tf.nn.relu(layer)

	return layer, weights, biases

def flaten_layer(layer):

	layer_shape = layer.get_shape()
	# Input shape is assumed to be as:
	#[num_images, img_height,img_width,num_channels]
	num_features = layer_shape[1:4].num_elements()
	flat_layer = tf.reshape(layer, [-1, num_features])

	return flat_layer, num_features

def fc_layer(img, num_inputs, num_outputs, relu, is_train):  # Use Rectified Linear Unit (ReLU)?
	shape = [num_inputs, num_outputs]

	if is_train == True:
		with tf.variable_scope("w_and_b"):
			weights = tf.get_variable(
				"weights", shape, \
				initializer=tf.random_normal_initializer(0, 0.01))
			biases = tf.get_variable(
			    "biases", [num_outputs], \
			    initializer=tf.constant_initializer(0.0))
	else:
		with tf.variable_scope("w_and_b",reuse = True):
			weights = tf.get_variable("weights")
			biases = tf.get_variable("biases")

	layer = tf.matmul(img, weights) + biases

	if relu:
		layer = tf.nn.relu(layer)

	return layer, weights, biases
####################################################################
# Convolutional layers and Full connected layer sizes
#  Convolutional layer 1
filter_size1 = 5
num_channels1 = 20

# Convolutional layer 2
filter_size2 = 7
num_channels2 = 50

# Convolutional layer 3
filter_size3 = 10
num_channels3 = 500


CASE = 2
logs_path = '/home/tjdias/Desktop/py_multimodal/tensorflow_logs/example'

timer = time.time()

tf_train_dataset = tf.placeholder(tf.float32, \
	shape=[None, imgSize, imgSize, num_channels])
tf_train_labels = tf.placeholder(tf.float32, \
	shape=[None, num_classes])

tf_test_dataset = tf.placeholder(tf.float32,\
	shape=[None, imgSize, imgSize, num_channels])
tf_test_labels = tf.placeholder(tf.float32, \
	shape=[None, num_classes])

def model(x, is_train):
	with tf.variable_scope("conv1"):
		conv_layer1, weights1, biases1 = convolution_layer(
			x, num_channels, filter_size1, num_channels1,\
			 True, False, is_train)
		variable_summaries(weights1, 'conv1' + '/weights')
		variable_summaries(weights1, 'conv1' + '/biases')
		# tf.scalar_summary("biases1", biases1)
	# print("Conv1 layer:", conv_layer1)

	with tf.variable_scope("conv2"):
		conv_layer2,weights2, biases2 = convolution_layer(
			conv_layer1, num_channels1, filter_size2, num_channels2, \
			True, False, is_train)
		variable_summaries(weights2, 'conv2' + '/weights')
		variable_summaries(weights2, 'conv2' + '/biases')
		# tf.scalar_summary("weights2", weights2)
		# tf.scalar_summary("biases2", biases2)
	# print("Conv2 layer:", conv_layer2)

	with tf.variable_scope("conv3"):
		conv_layer3, weights3, biases3 = convolution_layer(
			conv_layer2, num_channels2, filter_size3, num_channels3, \
			False, True, is_train)
		variable_summaries(weights2, 'conv3' + '/weights')
		variable_summaries(weights2, 'conv3' + '/biases')
		# tf.scalar_summary("weights3", weights3)
		# tf.scalar_summary("biases3", biases3)
	# print("Conv3 layer:", conv_layer3)

	with tf.variable_scope("flat1"):
		flat_layer, num_features = flaten_layer(conv_layer3)
	# print("Flat layer:", flat_layer)

	with tf.variable_scope("fcon1"):
		fc_layer1, weights4, biases4 = fc_layer(
		 	flat_layer, num_features, num_classes, False, is_train)
		variable_summaries(weights2, 'fc1' + '/weights')
		variable_summaries(weights2, 'fc1' + '/biases')
		# tf.scalar_summary("weights4", weights4)
		# tf.scalar_summary("biases4", biases4)
	return (fc_layer1)

# Training computation.
logits = model(tf_train_dataset, True)

train_prediction = tf.nn.softmax(logits)
test_prediction = tf.nn.softmax(model(tf_test_dataset, False))
# test_prediction = tf.nn.softmax(model(tf_test_dataset, False))

loss = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
tf.scalar_summary("loss", loss)

optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)

correct_prediction = tf.equal(tf.argmax(train_prediction, 1),\
	tf.argmax(tf_train_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
tf.scalar_summary("accuracy_train", accuracy)

correct_prediction_test = tf.equal(tf.argmax(test_prediction, 1),\
	tf.argmax(tf_test_labels, 1))
accuracy_test = tf.reduce_mean(tf.cast(correct_prediction_test,tf.float32))

#############################################################################
# tf.scalar_summary("accuracy_test",accuracy_test)
# Merge all summaries into a single op
merged_summary_op = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter(logs_path + '/train', sess.graph)
valid_writer = tf.train.SummaryWriter(logs_path + '/valid')
test_writer = tf.train.SummaryWriter(logs_path + '/test')
#
#############################################################################
num_epochs = 10 #10
batch_size = 100

with tf.Session() as session:
	session.run(tf.initialize_all_variables())
	summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
	
	for epoch in range(num_epochs):
		avg_cost = 0
		train_pred = 0
		num_steps = int(nmbTrainImg / batch_size)

		for step in range(num_steps):
			offset = (step * batch_size) % (trainClass.shape[0] - batch_size)

			if CASE == 1:
				batch_data = trainImg[offset:(offset + batch_size), :, :, :]
			else:
				batch_data = trainMask[offset:(offset + batch_size), :, :, :]

			batch_labels = trainClass[offset:(offset + batch_size), :]
			feed_dict_train = {tf_train_dataset: batch_data,\
			 tf_train_labels: batch_labels}
			
			_, l, predictions, summary = session.run(
				[optimizer, loss, train_prediction, merged_summary_op],\
				 feed_dict=feed_dict_train)
			# valid = accuracy.run(.eval(), validClass)
			summary_writer.add_summary(summary, \
				epoch * num_steps + step)
			
			avg_cost += l / num_steps
			train_pred = accuracy.eval(feed_dict = feed_dict_train)

		if CASE ==1:
			feed_dict_valid = {tf_test_dataset: validImg, \
			tf_test_labels : validClass}
		else:
			feed_dict_valid = {tf_test_dataset: validMask, \
			tf_test_labels : validClass}

		valid = accuracy_test.eval(feed_dict = feed_dict_valid)
		print("Epoch:", '%d' % (epoch+1),\
		 "Train loss=", "{:.3f}".format(avg_cost),\
		 "Train Accuracy=", "{:.3f}".format(train_pred),\
		 "Valid Accuracy=", "{:.3f}".format(valid))
	# print("Optimization Finished!")
	if CASE == 1:
		feed_dict_test = {tf_test_dataset: testImg, tf_test_labels : testClass}
	else:
		feed_dict_test = {tf_test_dataset: testMask, tf_test_labels : testClass}
	
	print('Test accuracy: %.3f%%' % accuracy_test.eval(feed_dict = feed_dict_test))
	print("Elapsed time is " + str(time.time() - timer) + " seconds.")
