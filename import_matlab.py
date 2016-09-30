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
trainImgAux = mat_contents['trainImg']
trainMaskAux = mat_contents['trainMask']
trainClass = mat_contents['trainClass']
# print(trainImg.shape)
# print(trainMask.shape)
# print(trainClass.shape)

mat_contents = sio.loadmat('Val_stuff.mat')
# print(mat_contents.keys())
validImgAux = mat_contents['valImg']
validMaskAux = mat_contents['valMask']
validClass = mat_contents['valClass']
# print(validImg.shape)
# print(validMask.shape)
# print(validClass.shape)

mat_contents = sio.loadmat('Tst_stuff.mat')
# print(mat_contents.keys())
testImgAux = mat_contents['testImg']
testMaskAux = mat_contents['testMask']
testClass = mat_contents['testClass']	


trainClass=trainClass-1
validClass=validClass-1
testClass=testClass-1

print (trainImgAux.shape)
print(testImgAux.shape[2])
print(testClass.shape)

trainImg = np.zeros(
	[trainImgAux.shape[2],imgSize,imgSize,num_channels],dtype = 'float32')
trainMask = np.zeros(
	[trainMaskAux.shape[2],imgSize,imgSize,num_channels],dtype = "float32")

validImg = np.zeros(
	[validImgAux.shape[2],imgSize,imgSize,num_channels],dtype = 'float32')
validMask = np.zeros(
	[validMaskAux.shape[2],imgSize,imgSize,num_channels],dtype = "float32")

testImg = np.zeros(
	[testImgAux.shape[2],imgSize,imgSize,num_channels],dtype = 'float32')
testMask = np.zeros(
	[testMaskAux.shape[2],imgSize,imgSize,num_channels],dtype = "float32")

print(trainImg.shape,trainMask.shape)
print(validImg.shape,validMask.shape)
print(testImg.shape,testMask.shape)

# Z = misc.toimage(trainMaskAux[:,:,501])       # Create a PIL image
# plt.imshow(Z, cmap='gray')
# plt.xlabel(trainClass[0,500])
# plt.show()

for idx1 in range(nmbTrainImg):
	trainImg[idx1,:,:,0]=trainImgAux[:,:,idx1]
	trainMask[idx1,:,:,0]=trainMaskAux[:,:,idx1]

for idx2 in range(nmbValImg):
	validImg[idx2,:,:,0]=validImgAux[:,:,idx2]
	validMask[idx2,:,:,0]=validMaskAux[:,:,idx2]

for idx3 in range(nmbTestImg):
	testImg[idx3,:,:,0]=testImgAux[:,:,idx3]
	testMask[idx3,:,:,0]=testMaskAux[:,:,idx3]

# print (trainImg.shape)
# Z = misc.toimage(trainMask[501,:,:,0])       # Create a PIL image
# plt.imshow(Z, cmap='gray')
# plt.xlabel(testClass[0,500])
# plt.show()

# print(type(trainClass))
# print(trainClass.shape)
# print(trainImg[0,:,:,:])

trainClass = np.squeeze(np.asarray(trainClass))
validClass = np.squeeze(np.asarray(validClass))
testClass = np.squeeze(np.asarray(testClass))

# print(trainClass.shape)
# print(validClass.shape)
# print(testClass.shape)
print(testClass[500])
trainClass = (np.arange(num_classes) == trainClass[:,None]).astype(np.float32)
validClass = (np.arange(num_classes) == validClass[:,None]).astype(np.float32)
testClass = (np.arange(num_classes) == testClass[:,None]).astype(np.float32)

print("Fiz um novo commasddasdit")
print(testClass[500])

# print(trainClass.shape)
# print(validClass.shape)
# print(testClass.shape)
# #######################################################################
#TensorFlow Graph

def convolution_layer(
	input, num_input_channels, filter_size, num_filters, use_pooling,use_relu):
	# print (filter_size)
	shape = [filter_size,filter_size,num_input_channels, num_filters]

	weights = tf.get_variable(
		"weights", shape, initializer=tf.random_normal_initializer(0,0.01))
    # Create variable named "biases".
	biases = tf.get_variable("biases", [num_filters], initializer=tf.constant_initializer(0.0))

	layer = tf.nn.conv2d(input, weights,[1,1,1,1],'VALID') + biases
	# print(layer)

	if use_pooling:
		layer = tf.nn.max_pool(layer, [1,2,2,1],[1,2,2,1],'SAME')

	if use_relu:
		layer = tf.nn.relu(layer)

	# return weights,biases
	return layer,weights

def flaten_layer(layer):

	layer_shape = layer.get_shape()
	#Input shape is assumed to be as:
	#[num_images, img_height,img_width,num_channels]
	num_features = layer_shape[1:4].num_elements()
	flat_layer = tf.reshape(layer, [-1, num_features])

	return flat_layer,num_features

def fc_layer(img,num_inputs, num_outputs, relu): # Use Rectified Linear Unit (ReLU)?

	shape = [num_inputs, num_outputs]

	weights = tf.get_variable(
		"weights", shape, initializer=tf.random_normal_initializer(0,0.01))
	biases = tf.get_variable("biases", [num_outputs], initializer=tf.constant_initializer(0.0))

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
	layer = tf.matmul(img, weights) + biases

    # Use ReLU?
	if relu:
		layer = tf.nn.relu(layer)

	return layer

#Convolutional layers and Full connected layer sizes
#Convolutional layer 1
filter_size1 = 5
num_channels1 = 20

#Convolutional layer 2
filter_size2 = 7
num_channels2 = 50

#Convolutional layer 3		
filter_size3 =10
num_channels3 = 500

#Convolutional layer 4
filter_size4 =1
num_channels4 = 500


CASE = 1
# tf_train_dataset = tf.placeholder(
#     tf.float32, shape=(batch_size, image_size, image_size, num_channels))
tf_train_dataset = tf.placeholder(tf.float32,  shape=[
	None, imgSize, imgSize, num_channels])
tf_train_labels = tf.placeholder(
	tf.float32, shape = [None, num_classes])

if CASE == 1:
	tf_valid_dataset = tf.constant(validImg)
	# tf_test_dataset = tf.constant(testImg)
	tf_test_dataset = tf.constant(testImg)
else :
	tf_valid_dataset = tf.constant(validMask)
	# tf_test_dataset = tf.constant(testImg)
	tf_test_dataset = tf.constant(testMask)


print(tf_train_dataset)
print (tf_valid_dataset)
print (tf_test_dataset)

def model(x):
	with tf.variable_scope("conv1"):
		conv_layer1,conv_weights1 = convolution_layer(
			x,num_channels,filter_size1, num_channels1, True,False)
	print("Conv1 layer:", conv_layer1)
	
	with tf.variable_scope("conv2"):
		conv_layer2,conv_weights2 = convolution_layer(
			conv_layer1,num_channels1,filter_size2, num_channels2, True,False)
	
	print("Conv2 layer:", conv_layer2)
	
	with tf.variable_scope("conv3"):
		conv_layer3,conv_weights3 = convolution_layer(
			conv_layer2,num_channels2,filter_size3, num_channels3, False,True)
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
	logits = model (tf_train_dataset)
	scope.reuse_variables()
	valid_logits = model(tf_valid_dataset)
	test_logits = model(tf_test_dataset)
	
	
loss = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(logits,tf_train_labels))    

optimizer = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)

# Predictions for the training, validation, and test data.
train_prediction = tf.nn.softmax(logits)
valid_prediction = tf.nn.softmax(valid_logits)
test_prediction = tf.nn.softmax(test_logits)


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

#Placeholder variables
num_steps = 100
batch_size = 100

with tf.Session() as session:
  session.run(tf.initialize_all_variables())
  print('Initialized')
  for step in range(num_steps):
    # offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # batch_data = trainImg[:100,:,:]
    # batch_labels = trainClass[:100,:]
    offset = (step * batch_size) % (trainClass.shape[0] - batch_size)
    # batch_data = trainImg[:100, :, :, :]
    # batch_labels = trainClass[:100, :] 
    if CASE == 1:
    	batch_data = trainImg[offset:(offset + batch_size), :, :, :]
    else:
    	batch_data = trainMask[offset:(offset + batch_size), :, :, :]

    batch_labels = trainClass[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
    	[optimizer, loss, train_prediction], feed_dict=feed_dict)
    if step%10==0:
	    print('Minibatch loss at step %d: %f' % (step, l))
	    print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
	    print('Validation accuracy: %.1f%%' % accuracy(
         	valid_prediction.eval(), validClass))

  # print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), testClass)) 
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), testClass))