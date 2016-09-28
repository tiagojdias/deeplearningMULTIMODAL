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
from bwmorph_thin import bwmorph_thin


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

#############################################
def fgaussian(sizew, sizeh, sigma):
     m = sizew
     n = sizeh
     h, k = m // 2, n // 2
     x, y = np.mgrid[-h:h + 1, -k:k + 1]
     return (np.exp(-(x**2 + y**2) / (2 * sigma**2)))

h = fgaussian(5,5,1)
# print (h.shape)
# define the function blocks
def no_cancer():
	segMask = np.zeros(shape=(imgSize, imgSize), dtype=np.float32);
	return (segMask)

def benign():
	tumorSize = 10
	segMask = np.zeros(shape=(imgSize, imgSize), dtype=np.float32);

	
	posX = min(imgSize-tumorSize,max(
			1,round(random.random()*(imgSize - tumorSize))));
	posY = min(imgSize-tumorSize,max(
			1,round(random.random()*(imgSize - tumorSize))));

	segMask[posY:posY+math.ceil(
		random.random()*tumorSize),posX:posX + math.ceil(
		random.random()*tumorSize)]=1.0
	
	return (segMask)

def malign():
	minTumorSize = 10
	tumorSize = 40
	segMask = np.zeros(shape=(imgSize, imgSize), dtype=np.float32);

	posX = min(imgSize-tumorSize,max(
			1,round(random.random()*(imgSize - tumorSize))));
	posY = min(imgSize-tumorSize,max(
			1,round(random.random()*(imgSize - tumorSize))));

	segMask[posY:posY+max(
		minTumorSize,math.ceil(
		random.random()*tumorSize)),posX:posX + max(
		minTumorSize,math.ceil(random.random()*tumorSize))]=1.0
	

	return (segMask)

# map the inputs to the function blocks
options = {0 : no_cancer,
           1 : benign,
		   2 : malign
}

trainImg= np.zeros(shape=(imgSize,imgSize,nmbTrainImg))
trainMask= np.zeros(shape=(imgSize,imgSize,nmbTrainImg))
trainClass= np.zeros(nmbTrainImg).astype(np.int)

validImg= np.zeros(shape=(imgSize,imgSize,nmbValImg))
validMask= np.zeros(shape=(imgSize,imgSize,nmbValImg))
validClass= np.zeros(nmbValImg).astype(np.int)

testImg= np.zeros(shape=(imgSize,imgSize,nmbTestImg))
testMask= np.zeros(shape=(imgSize,imgSize,nmbTestImg))
testClass= np.zeros(nmbTestImg).astype(np.int)

for itr in range(nmbTrainImg + nmbValImg + nmbTestImg):
	img = 10 * np.random.randn(imgSize, imgSize) + 100

	# print(type(img), img.shape)

	img=convolve2d(img, h, 'same')
	# img = ndimage.filters.gaussian_filter(img, sigma=2.0)

	tmp = random.random() < np.cumsum(priorClasses)

	cls = np.min(np.where(tmp == True))
	# print (cls)

	segMask = options[cls]()
	
	img = img + 50*segMask;
	# Z = misc.toimage(data)       # Create a PIL image
	# plt.imshow(Z, cmap='gray', interpolation='nearest')
	# plt.show()
	
    #         % no noise in segmentation
    # segMask(posY:posY+ceil(rand*tumorSize),posX:posX+ceil(rand*tumorSize)) = 1;
	# Z = misc.toimage(img)       # Create a PIL image
	# plt.imshow(Z, cmap='gray')
	# plt.show()


	img = np.maximum(
		np.zeros(shape=(imgSize, imgSize), dtype=np.float32), img + 10*np.random.randn(
			imgSize,imgSize) + 30)

	img = np.minimum(
		255 + np.zeros(shape=(imgSize, imgSize), dtype=np.float32),img)

	# img = np.minimum(140 +np.zeros(shape=(imgSize, imgSize),img)

	if (cls == 1) | (cls == 2):
		auxImg = np.greater(
			img, 140 +np.zeros(shape=(imgSize, imgSize), dtype=np.float32))
		
		segMask = np.multiply(segMask,auxImg)

	else:
		segMask = np.greater(
			img, 140 +np.zeros(shape=(imgSize, imgSize), dtype=np.float32))

	segMask = ndimage.morphology.binary_opening(segMask)
	segMask = ndimage.morphology.binary_closing(segMask)
	# segMask = bwmorph_thin(segMask)
	segMask = np.multiply(segMask,255 +np.zeros(
		shape=(imgSize, imgSize), dtype=np.float32))

	if itr < nmbTrainImg:
		trainImg[:,:,itr]=img
		trainMask[:,:,itr]=segMask
		trainClass[itr] = cls;
		# print ("train")
	elif itr < nmbTrainImg + nmbValImg:
		validImg[:,:,itr-nmbTrainImg]=img
		validMask[:,:,itr-nmbTrainImg]=segMask
		validClass[itr-nmbTrainImg] = cls;
		# print("valid")
	elif itr < nmbTrainImg + nmbValImg + nmbTestImg:
		testImg[:,:,itr-(nmbTrainImg + nmbValImg)]=img
		testMask[:,:,itr-(nmbTrainImg + nmbValImg)]=segMask
		testClass[itr-(nmbTrainImg + nmbValImg)] = cls;
		# print("Test")
	# Z = misc.toimage(segMask)       # Create a PIL image
	# plt.imshow(Z, cmap='gray')
	# plt.show()		

# print (trainImg.shape)
CASE = 1

if CASE == 1:
	trainImg = trainImg.reshape(
		(-1,imgSize,imgSize,num_channels)).astype(np.float32)
	validImg = validImg.reshape(
		(-1,imgSize,imgSize,num_channels)).astype(np.float32)
	testImg = testImg.reshape(
		(-1,imgSize,imgSize,num_channels)).astype(np.float32)
else:
	trainImg = trainMask.reshape(
		(-1,imgSize,imgSize,num_channels)).astype(np.float32)
	validImg = validMask.reshape(
		(-1,imgSize,imgSize,num_channels)).astype(np.float32)
	testImg = testMask.reshape(
		(-1,imgSize,imgSize,num_channels)).astype(np.float32)

print(type(trainClass))
print(trainClass.shape)
trainClass = (np.arange(
	num_classes) == trainClass[:,None]).astype(np.float32)
validClass = (np.arange(
	num_classes) == validClass[:,None]).astype(np.float32)
testClass = (np.arange(
	num_classes) == testClass[:,None]).astype(np.float32)

print(trainClass.shape)


# # print (testImg.shape)
# #######################################################################
#TensorFlow Graph
def variable_weights(size_w):
	weights = tf.Variable(tf.truncated_normal(size_w, stddev= 0.05))
	return weights

def variable_biases(size_b):
	biases = tf.Variable(tf.zeros([size_b]))
	return biases

def convolution_layer(
	input, num_input_channels, filter_size, num_filters, use_pooling,use_relu):
	# print (filter_size)
	shape = [filter_size,filter_size,num_input_channels, num_filters]

	weights = variable_weights(shape)
	biases = variable_biases(num_filters)

	layer = tf.nn.conv2d(input, weights,[1,1,1,1],'VALID') + biases
	# print(layer)

	if use_pooling:
		layer = tf.nn.max_pool(layer, [1,2,2,1],[1,2,2,1],'VALID')

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

def fc_layer(input,num_inputs, num_outputs, use_relu): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = variable_weights([num_inputs, num_outputs])
    biases = variable_biases(num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

#Convolutional layers and Full connected layer sizes
#Convulocional layer 1
filter_size1 = 5
num_channels1 = 20

#Convulocional layer 2
filter_size2 = 7
num_channels2 = 50

#Convulocional layer 3
filter_size3 =10
num_channels3 = 500

#Convulocional layer 4
filter_size4 =1
num_channels4 = 500

#Placeholder variables
num_steps = 10
batch_size = 100
# tf_train_dataset = tf.placeholder(
#     tf.float32, shape=(batch_size, image_size, image_size, num_channels))
tf_train_dataset = tf.placeholder(tf.float32,  shape=(
	None, imgSize, imgSize, num_channels))
tf_train_labels = tf.placeholder(tf.float32, shape = [None, num_classes])

tf_valid_dataset = tf.constant(validImg)
tf_test_dataset = tf.constant(testImg)

def model(x):
	conv_layer1,conv_weights1 = convolution_layer(
		x,num_channels,filter_size1, num_channels1, True,False)
	print (conv_layer1)
	conv_layer2,conv_weights2 = convolution_layer(
		conv_layer1,num_channels1,filter_size2, num_channels2, True,False)
	print (conv_layer2)
	conv_layer3,conv_weights3 = convolution_layer(
		conv_layer2,num_channels2,filter_size3, num_channels3, False,True)
	print (conv_layer3)
	conv_layer4,conv_weights4 = convolution_layer(
		conv_layer3,num_channels3,filter_size4, num_classes, False,False)
	print (conv_layer4) 
	final_layer, num_features = flaten_layer(conv_layer4)
	return (final_layer)

# Training computation.
logits = model (tf_train_dataset)

loss = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(logits,tf_train_labels))    

optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# Predictions for the training, validation, and test data.
train_results = tf.nn.softmax(logits)
valid_results = tf.nn.softmax(model(tf_valid_dataset))
test_results = tf.nn.softmax(model(tf_test_dataset))

# correct_prediction = tf.equal(train_pred_cls,train_true_cls)
# accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


with tf.Session() as session:
  session.run(tf.initialize_all_variables())
  print('Initialized')
  for step in range(num_steps):
    # offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # batch_data = trainImg[:100,:,:]
    # batch_labels = trainClass[:100,:]
    offset = (step * batch_size) % (trainClass.shape[0] - batch_size)
    batch_data = trainImg[offset:(offset + batch_size), :, :, :]
    batch_labels = trainClass[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
    	[optimizer, loss, train_results], feed_dict=feed_dict)
    
    print('Minibatch loss at step %d: %f' % (step, l))
    print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
    print('Validation accuracy: %.1f%%' % accuracy(
        valid_results.eval(), validClass))

  print('Test accuracy: %.1f%%' % accuracy(test_results.eval(), testClass)) 