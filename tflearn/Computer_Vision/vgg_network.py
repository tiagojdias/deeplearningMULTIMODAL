from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, flatten,fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import tflearn.datasets.oxflower17 as oxflower17
X, Y = oxflower17.load_data(one_hot = True)

network = input_data(shape = [None, 224, 224, 3])

network = conv_2d(network, 64, 3, activation = 'relu')
network = conv_2d(network, 64, 3, activation = 'relu')
network = max_pool_2d(network, 2, strides = 2)

network = conv_2d(network, 128, 3, activation = 'relu')
network = conv_2d(network, 128, 3, activation = 'relu')
network = max_pool_2d(network, 2, strides = 2)

network = conv_2d(network, 256, 3, activation = 'relu')
network = conv_2d(network, 256, 3, activation = 'relu')
network = conv_2d(network, 256, 3, activation = 'relu')
network = max_pool_2d(network, 2, strides = 2)

network = conv_2d(network, 512, 3, activation = 'relu')
network = conv_2d(network, 512, 3, activation = 'relu')
network = conv_2d(network, 512, 3, activation = 'relu')
network = max_pool_2d(network, 2, strides = 2)

network = conv_2d(network, 512, 3, activation = 'relu')
network = conv_2d(network, 512, 3, activation = 'relu')
network = conv_2d(network, 512, 3, activation = 'relu')
network = max_pool_2d(network, 2, strides = 2)

network = fully_connected(network, 4096, activation = 'relu')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation = 'relu')
network = dropout(network, 0.5)
network = fully_connected(network, 17, activation = 'softmax')

network = regression(network, optimizer = 'rmsprop',\
	loss = 'categorical_crossentropy', learning_rate = 0.001)

model = tflearn.DNN(network, checkpoint_path = 'model_vgg', 
	max_checkpoints = 1, tensorboard_dir = '/home/tjdias/tflearn_logs/',\
	tensorboard_verbose = 0)

model.fit(X, Y, n_epoch = 500, shuffle = True,
	show_metric = True, batch_size = 32, snapshot_step = 500,\
	snapshot_epoch = False, run_id ='vgg_oxflowes17')