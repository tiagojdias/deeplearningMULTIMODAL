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

for idx3 in range(nmbTestImg - 100):
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
def convolution_layer(
	input, num_input_channels, filter_size, num_filters, use_pooling, use_relu):
	# print (filter_size)
	shape = [filter_size, filter_size, num_input_channels, num_filters]

	weights = tf.get_variable(
		"weights", shape, initializer=tf.random_normal_initializer(0, 0.01))
    # Create variable named "biases".
	biases = tf.get_variable(
	    "biases", [num_filters], initializer=tf.constant_initializer(0.0))

	layer = tf.nn.conv2d(input, weights, [1, 1, 1, 1], 'VALID') + biases
	# print(layer)

	if use_pooling:
		layer = tf.nn.max_pool(layer, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

	if use_relu:
		layer = tf.nn.relu(layer)

	# return weights,biases
	return layer, weights
def flaten_layer(layer):

	layer_shape = layer.get_shape()
	# Input shape is assumed to be as:
	#[num_images, img_height,img_width,num_channels]
	num_features = layer_shape[1:4].num_elements()
	flat_layer = tf.reshape(layer, [-1, num_features])

	return flat_layer, num_features
def fc_layer(img, num_inputs, num_outputs, relu):  # Use Rectified Linear Unit (ReLU)?

	shape = [num_inputs, num_outputs]

	weights = tf.get_variable(
		"weights", shape, initializer=tf.random_normal_initializer(0, 0.01))
	biases = tf.get_variable(
	    "biases", [num_outputs], initializer=tf.constant_initializer(0.0))

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
	layer = tf.matmul(img, weights) + biases

    # Use ReLU?
	if relu:
		layer = tf.nn.relu(layer)

	return layer

# Convolutional layers and Full connected layer sizes
# Convolutional layer 1
filter_size1 = 5
num_channels1 = 20

# Convolutional layer 2
filter_size2 = 7
num_channels2 = 50

# Convolutional layer 3
filter_size3 = 10
num_channels3 = 500

# Convolutional layer 4
filter_size4 = 1
num_channels4 = 500


CASE = 1
# tf_train_dataset = tf.placeholder(
#     tf.float32, shape=(batch_size, image_size, image_size, num_channels))
timer = time.time()

tf_train_dataset = tf.placeholder(tf.float32, shape=[None, imgSize, imgSize, num_channels])
tf_train_labels = tf.placeholder(tf.float32, shape=[None, num_classes])

if CASE == 1:
	tf_valid_dataset = tf.constant(validImg)
	# tf_test_dataset = tf.constant(testImg)
	tf_test_dataset = tf.constant(testImg)
else:
	tf_valid_dataset = tf.constant(validMask)
	# tf_test_dataset = tf.constant(testImg)
	tf_test_dataset = tf.constant(testMask)


print(tf_train_dataset)
print (tf_valid_dataset)
print (tf_test_dataset)


def model(x):
	with tf.variable_scope("conv1"):
		conv_layer1, conv_weights1 = convolution_layer(
			x, num_channels, filter_size1, num_channels1, True, False)
	print("Conv1 layer:", conv_layer1)

	with tf.variable_scope("conv2"):
		conv_layer2, conv_weights2 = convolution_layer(
			conv_layer1, num_channels1, filter_size2, num_channels2, True, False)

	print("Conv2 layer:", conv_layer2)

	with tf.variable_scope("conv3"):
		conv_layer3, conv_weights3 = convolution_layer(
			conv_layer2, num_channels2, filter_size3, num_channels3, False, True)
	print("Conv3 layer:", conv_layer3)

	# with tf.variable_scope("conv4"):
	# 	conv_layer4,conv_weights4 = convolution_layer(
	# 		conv_layer3,num_channels3,filter_size4, num_classes, False,False)
	# print (conv_layer4)

	with tf.variable_scope("flat1"):
		flat_layer, num_features = flaten_layer(conv_layer3)

	print("Flat layer:", flat_layer)

	with tf.variable_scope("fcon1"):
		fc_layer1 = fc_layer(
		 	flat_layer, num_features, num_classes, False)
	print (fc_layer1)
	# final_layer = fc_layer(
	#  	flat_layer, num_channels3, num_classes, False)
	# print("Final layer:",final_layer)
	# return (flat_layer)
	return (fc_layer1)

# Training computation.
with tf.variable_scope("my_model") as scope:
	logits = model(tf_train_dataset)
	scope.reuse_variables()
	# valid_logits = model(tf_valid_dataset) 
	test_logits = model(tf_test_dataset)
	
loss = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)

# Predictions for the training, validation, and test data.
train_prediction = tf.nn.softmax(logits)
# valid_prediction = tf.nn.softmax(valid_logits)
test_prediction = tf.nn.softmax(test_logits)


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

# Placeholder variables
num_epochs = 10 #10
batch_size = 100

#############################################################################
logs_path = '/tmp/tensorflow_logs/example'
# Create a summary to monitor cost tensor
tf.scalar_summary("loss", loss)
# #Create a summary to monitor accuracy tensor
# tf.scalar_summary("accuracy", acc)
# Merge all summaries into a single op
merged_summary_op = tf.merge_all_summaries()
#############################################################################
x = []
y = []
z = []
fig,axs = plt.subplots()
axs.set_xlim([1,10])

with tf.Session() as session:
	session.run(tf.initialize_all_variables())
	tf.train.import_meta_graph("/tmp/my-model-10000.meta")
	hparams = tf.get_collection("hparams")
	print (hparams)
	# summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
	
	# for epoch in range(num_epochs):
	# 	avg_cost = 0
	# 	num_steps = int(nmbTrainImg / batch_size)

	# 	for step in range(num_steps):
	# 		offset = (step * batch_size) % (trainClass.shape[0] - batch_size)

	# 		if CASE == 1:
	# 			batch_data = trainImg[offset:(offset + batch_size), :, :, :]
	# 		else:
	# 			batch_data = trainMask[offset:(offset + batch_size), :, :, :]

	# 		batch_labels = trainClass[offset:(offset + batch_size), :]
	# 		feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
	# 		_, l, predictions, summary = session.run(
	# 			[optimizer, loss, train_prediction, merged_summary_op], feed_dict=feed_dict)
	# 		# valid = accuracy.run(valid_prediction.eval(), validClass)
	# 		summary_writer.add_summary(summary, epoch * num_steps + step)
	# 		avg_cost += l / num_steps
		
	# 	x.append(epoch+1)
	# 	y.append(avg_cost)
	# 	# z.append(5)
	# 	axs.plot(x,y,'-b')
	# 	# axs.plot(x,z,'-r')
	# 	# axs.set_ylim([,10])
	# 	# print (max(y))
	# 	plt.ylim(0,max(y)+1)
	# 	plt.xlabel('Epochs')
	# 	plt.title('Average Loss')
	# 	plt.grid('on')
	# 	plt.pause(0.0001)

	# 	print("Epoch:", '%d' % (epoch+1), "Train loss=", "{:.3f}".format(avg_cost))
	# 	valid = accuracy(valid_prediction.eval(), validClass)
	# 	print("Epoch:", '%d' % (epoch+1), "Valid Accuracy=", "{:.3f}".format(valid))
	# # print("Optimization Finished!")
	# print('Test accuracy: %.3f%%' % accuracy(test_prediction.eval(), testClass))
	print("Elapsed time is " + str(time.time() - timer) + " seconds.")
