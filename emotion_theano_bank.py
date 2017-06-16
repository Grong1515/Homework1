# coding: utf-8

import xml.etree.ElementTree as ElementTree
#from sklearn.linear_model.logistic import LogisticRegression
#from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import numpy as np
import os

os.environ["THEANO_FLAGS"] = "exception_verbosity=high"

import theano
import theano.tensor as T
import lasagne
from sklearn.metrics import recall_score, precision_score, f1_score
import gensim
import string
import re
import time
import pickle
from random import shuffle


def xml_parse (filename):
	
	e = ElementTree.parse(filename).getroot()
	i = 0

	text_list = list()
	results_list = list()

	while i < (len(e[1])):
		results_local = list()
		n = 4
		while n < len(e[1][i]):
			if e[1][i][n].text != 'NULL':
				results_local.append(int(e[1][i][n].text))
			n = n + 1
		if len(results_local) == 1:
			results_list.extend(results_local)
			text_list.append(e[1][i][3].text)
		i = i + 1

	return results_list, text_list



def load_dataset(input_file_train, input_file_test):

	results_list_train, text_list_train = xml_parse(input_file_train)
	results_list_train_shuf = []
	text_list_train_shuf = []
	index_shuf = range(len(results_list_train))
	shuffle(list(index_shuf))
	for i in index_shuf:
		results_list_train_shuf.append(results_list_train[i])
		text_list_train_shuf.append(text_list_train[i])
	
	results_list_train, text_list_train = results_list_train_shuf, text_list_train_shuf
	
	sentences = [[word for word in re.findall(r"[\w']+|[.,!?;_]", document)] for document in text_list_train]
	max_sent_len = 0
	for sentence in sentences:
		if len(sentence) > max_sent_len:
			max_sent_len = len(sentence)
	model = gensim.models.Word2Vec(sentences, min_count=1)
	#model = gensim.models.KeyedVectors.load_word2vec_format('all.norm-sz100-w10-cb0-it1-min100.w2v', binary=True, unicode_errors='ignore')
	model.init_sims(replace=True)
	X_train = np.zeros((len(results_list_train), 1,  max_sent_len, len(model[','])), dtype=np.float32)
	n=0
	for i in range(len(results_list_train)):
		sentence = np.zeros((max_sent_len, len(model[','])), dtype=np.float32)
		for j, word in enumerate(sentences[i]):
			try:
				sentence[j] = model[word.lower()]
			except KeyError:
				#print(word)
				n += 1
				sentence[j] = 0
		X_train[i][0] = sentence
	
	#print(n)
	y_train = np.array(results_list_train, dtype=np.int32) + 1

	results_list_test, text_list_test = xml_parse(input_file_test)
	results_list_test_true = []
	text_list_test_true = []
	for i in range(len(results_list_test)):
		if results_list_train[i] != 0:
			results_list_test_true.append(results_list_test[i])
			text_list_test_true.append(text_list_test[i])
	
	results_list_test, text_list_test = results_list_test_true, text_list_test_true
	
	X_train, X_val = X_train[:-1600], X_train[-1600:]
	y_train, y_val = y_train[:-1600], y_train[-1600:]
	
	X_test = np.zeros((len(results_list_test), 1, max_sent_len, len(model[','])), dtype=np.float32)
	
	#print(max_sent_len)
	
	for i in range(len(results_list_test)):
		sentence = np.zeros((max_sent_len, len(model[','])), dtype=np.float32)
		for j, word in enumerate(sentences[i]):
			#print(j, word, len(model[word]))
			try:
				sentence[j] = model[word.lower()]
			except KeyError:
				sentence[j] = 0
		X_test[i][0] = sentence

	y_test = np.array(results_list_test, dtype=np.int32) + 1
	
	#print(X_train.shape)
	
	return X_train, y_train, X_val, y_val, X_test, y_test

	
def build_cnn(input_var=None):
	# As a third model, we'll create a CNN of two convolution + pooling stages
	# and a fully-connected hidden layer in front of the output layer.

	# Input layer, as usual:
	network = lasagne.layers.InputLayer(shape=(None, 1, 36, 100), input_var=input_var)
	lasagne.layers.get_output_shape(network)
	# This time we do not apply input dropout, as it tends to work less well
	# for convolutional layers.

	# Convolutional layer with 32 kernels of size 5x5. Strided and padded
	# convolutions are supported as well; see the docstring.
	network = lasagne.layers.Conv2DLayer(network, num_filters=16, filter_size=(3, 100), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
	# Expert note: Lasagne provides alternative convolutional layers that
	# override Theano's choice of which implementation to use; for details
	# please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

	# Max-pooling layer of factor 2 in both dimensions:
	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3,1))

	# A fully-connected layer of 256 units with 50% dropout on its inputs:
	network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=30, nonlinearity=lasagne.nonlinearities.rectify)

	# And, finally, the 10-unit output layer with 50% dropout on its inputs:
	network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=3, nonlinearity=lasagne.nonlinearities.softmax)

	return network


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(num_epochs=100):
	
	# Load the dataset
	#input_file_train = input('Enter the name of the xml file with training data: ')
	#input_file_test = input('Enter the name of the xml file with test data: ')
	input_file_train = 'data/bank_train_2016.xml'
	input_file_test = 'data/banks_test_etalon.xml'
	print("Loading data...")
	X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(input_file_train, input_file_test)
	
	#quit()
	
	# Prepare Theano variables for inputs and targets
	input_var = T.tensor4('inputs')
	target_var = T.ivector('targets')

	# Create neural network model (depending on first command line parameter)
	print("Building model and compiling functions...")
	network = build_cnn(input_var)

	# Create a loss expression for training, i.e., a scalar objective we want
	# to minimize (for our multi-class problem, it is the cross-entropy loss):
	prediction = lasagne.layers.get_output(network)
	loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
	loss = loss.mean()
	# We could add some weight decay as well here, see lasagne.regularization.

	# Create update expressions for training, i.e., how to modify the
	# parameters at each training step. Here, we'll use Stochastic Gradient
	# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
	params = lasagne.layers.get_all_params(network, trainable=True)
	updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

	# Create a loss expression for validation/testing. The crucial difference
	# here is that we do a deterministic forward pass through the network,
	# disabling dropout layers.
	test_prediction = lasagne.layers.get_output(network, deterministic=True)
	test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
	test_loss = test_loss.mean()
	# As a bonus, also create an expression for the classification accuracy:
	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

	# Compile a function performing a training step on a mini-batch (by giving
	# the updates dictionary) and returning the corresponding training loss:
	train_fn = theano.function([input_var, target_var], loss, updates=updates)

	# Compile a second function computing the validation loss and accuracy:
	val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

	# Finally, launch the training loop.
	print("Starting training...")
	# We iterate over epochs:
	for epoch in range(num_epochs):
		# In each epoch, we do a full pass over the training data:
		train_err = 0
		train_batches = 0
		start_time = time.time()
		for batch in iterate_minibatches(X_train, y_train, 50, shuffle=False):
			#print(batch[0].shape)
			inputs, targets = batch
			train_err += train_fn(inputs, targets)
			train_batches += 1

		# And a full pass over the validation data:
		val_err = 0
		val_acc = 0
		val_batches = 0
		for batch in iterate_minibatches(X_val, y_val, 50, shuffle=False):
			inputs, targets = batch
			err, acc = val_fn(inputs, targets)
			val_err += err
			val_acc += acc
			val_batches += 1

		# Then we print the results for this epoch:
		print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
		print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
		print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
		print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

	# After training, we compute and print the test error:
	test_err = 0
	test_acc = 0
	test_batches = 0
	for batch in iterate_minibatches(X_test, y_test, 50, shuffle=False):
		inputs, targets = batch
		targets_true = []
		inputs_true = []
		index_shuf = range(len(inputs))
		shuffle(list(index_shuf))
		for i in index_shuf:
			targets_true.append(targets[i])
			inputs_true.append(inputs[i])
		err, acc = val_fn(inputs_true, targets_true)
		test_err += err
		test_acc += acc
		test_batches += 1
	print("Final results:")
	print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
	print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))
	'''
	print('Recall [macro, micro]:')
	print(recall_score(y_test, results, average='macro'), recall_score(y_test, results, average='micro'))
	print('Precision [macro, micro]:')
	print(precision_score(y_test, results, average='macro'), precision_score(y_test, results, average='micro'))
	print('F1 [macro, micro]:')
	print(f1_score(y_test, results, average='macro'), f1_score(y_test, results, average='micro'))

'''


if __name__ == '__main__':
	main()

