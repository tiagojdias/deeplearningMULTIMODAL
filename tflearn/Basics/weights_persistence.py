from __future__ import absolute_import, division, print_function

import tflearn

import tflearn.datasets.mnist as mnist

X, Y, testX, testY = mnist.load_data(one_hot= True)

input_layer = tflearn.input_data(shape =[None, 784], name = 'input')
dense1 = tflearn.fully_connected(input_layer, 128, name = 'dense1')
dense2 = tflearn.fully_connected(dense1, 256, name = 'dense2')
softmax = tflearn.fully_connected(dense2, 10, activation = 'softmax')
regression = tflearn.regression(softmax, optimizer = 'adam',
	learning_rate = 0.001, loss = 'categorical_crossentropy')

# model = tflearn.DNN(regression, checkpoint_path ='model.tfl.ckpt')
# odel.fit(X, Y, n_epoch = 1, validation_set = (testX, testY),
# 	show_metric = True, snapshot_epoch = True, snapshot_step = 500,\
# 	run_id = 'model_and_weights')

# model.save("model.tfl")
model = tflearn.DNN(regression, checkpoint_path='model.tfl.ckpt')

model.load("model.tfl")

dense1_vars = tflearn.variables.get_layer_variables_by_name('dense1')
print("Dense1 layer weights:")
print (model.get_weights(dense1_vars[1]))