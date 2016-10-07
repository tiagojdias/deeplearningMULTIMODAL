from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.estimator import regression

from tflearn.datasets import cifar10
(X,Y), (X_test,Y_test)= cifar10.load_data()
X, Y = shuffle(X,Y)

# print (Y.shape) #5000
Y = to_categorical(Y, 10)
# print (Y.shape) #(5000,10)
Y_test = to_categorical(Y_test, 10)

network = input_data(shape=[None, 32, 32, 3])
network = conv_2d(network, 192, 5, activation = 'relu')
network = conv_2d(network, 160, 1, activation = 'relu')
network = conv_2d(network, 96, 1, activation = 'relu')
network = max_pool_2d(network,3, strides = 2)
network = dropout(network, 0.5)
network = conv_2d(network, 192, 5, activation = 'relu')
network = conv_2d(network, 192, 1, activation = 'relu')
network = conv_2d(network, 192, 1, activation = 'relu')
network = avg_pool_2d(network, 3, strides = 2)
network = dropout(network, 0.5)
network = conv_2d(network, 192, 3, activation = 'relu')
network = conv_2d(network, 192, 1, activation = 'relu')
network = conv_2d(network, 10, 1, activation = 'relu')
network = avg_pool_2d(network, 8)
network = flatten(network)
network = regression(network, optimizer = 'adam',
	loss = 'softmax_categorical_crossentropy',\
	learning_rate = 0.001)

model = tflearn.DNN(network, tensorboard_dir='/home/tjdias/tflearn_logs/')
model.fit(X, Y, n_epoch = 50, shuffle = True, validation_set =(X_test, Y_test),
	show_metric = True, batch_size = 128, run_id = 'cifar10_net_in_net')