import sys
import os
import time
import re

# os.environ["THEANO_FLAGS"] = "exception_verbosity=high"

import numpy as np
import theano
import theano.tensor as T

import lasagne
import gensim

from random import shuffle
import xml.etree.ElementTree as ET

BUFFER = 100


BANK_TRAIN_BASE = 'data/bank_train_2016.xml'
BANK_TEST_BASE = 'data/banks_test_etalon.xml'
TKK_TRAIN_BASE = 'data/tkk_train_2016.xml'
TKK_TEST_BASE = 'data/tkk_test_etalon.xml'

BANK_LIST = ['rshb', 'alfabank', 'vtb', 'bankmoskvy', 'raiffeisen', 'sberbank', 'uralsib', 'gazprom']
TKK_LIST = ['beeline', 'mts', 'megafon', 'tele2', 'rostelecom', 'komstar', 'skylink']



def load_data(file, names):
    tree = ET.parse(file)
    database = tree.getroot().find('database')

    train_set = []
    answer_set = []

    for table in database.iter('table'):
        train_set.append(re.split(r"[\.:]*[\",!?;_ \(\)]+", table.find('./*[@name="text"]').text))
        answer = None
        for item in names:
            if answer is None:
                value = table.find('./*[@name="{}"]'.format(item)).text
                if value in ['-1', '1', '0']:
                    answer = int(value)
                    break
                # if value == '-1':
                #     answer = 'negative'
                #     break
                # elif value == '0':
                #     answer = 'neutral'
                #     break
                # elif value == '1':
                #     answer = 'positive'
                #     break
        answer_set.append(answer)

    mix = list(range(len(train_set)))
    shuffle(mix)
    return np.array(shuffle_set(train_set, mix)), np.array(shuffle_set(answer_set, mix))


def shuffle_set(set, mix):
    response = list(range(len(mix)))
    for i in list(range(len(mix))):
        response[mix[i]] = set[i]
    return response



def load_dataset(train_base, test_base, items_list):
    

    X, y = load_data(train_base, items_list)

    max_sent_len = 50
    # for sentence in X:
    #     if len(sentence) > max_sent_len:
    #         max_sent_len = len(sentence)
    # print(max_sent_len)
    model = gensim.models.Word2Vec(X, min_count=1)
    #model = gensim.models.KeyedVectors.load_word2vec_format('all.norm-sz100-w10-cb0-it1-min100.w2v', binary=True, unicode_errors='ignore')
    model.init_sims(replace=True)
    X_train = np.zeros((len(y), 1, max_sent_len, BUFFER), dtype=np.float32)
    n=0
    for i in range(len(y)):
        sentence = np.zeros((max_sent_len, BUFFER), dtype=np.float32)
        for j, word in enumerate(X[i]):
            try:
                sentence[j] = model[word.lower()]
            except KeyError:
                #print(word)
                n += 1
                sentence[j] = 0
        X_train[i][0] = sentence
    
    #print(n)
    y_train = np.array(y, dtype=np.int32) + 1
    print("Handling test set...")
    X, y = load_data(test_base, items_list)
    X_test = np.zeros((len(y), 1, max_sent_len, BUFFER), dtype=np.float32)
    n=0
    for i in range(len(y)):
        sentence = np.zeros((max_sent_len, BUFFER), dtype=np.float32)
        for j, word in enumerate(X[i]):
            try:
                sentence[j] = model[word.lower()]
            except KeyError:
                #print(word)
                n += 1
                sentence[j] = 0
        X_train[i][0] = sentence
    y_test = np.array(y, dtype=np.int32) + 1

    # X_train = X[:int(len(X)/4*3)]
    # y_train = y[:int(len(X)/4*3)]

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:int(len(X_train)/4*3)], X_train[int(len(X_train)/4*3):]
    y_train, y_val = y_train[:int(len(y_train)/4*3)], y_train[int(len(y_train)/4*3):]

    print(X_train.shape, X_test.shape)
    return X_train, y_train, X_val, y_val, X_test, y_test


# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.

def build_mlp(input_var=None):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 50, BUFFER),
                                     input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out


def build_custom_mlp(input_var=None, depth=2, width=800, drop_input=.2,
                     drop_hidden=.5):
    # By default, this creates the same network as `build_mlp`, but it can be
    # customized with respect to the number and size of hidden layers. This
    # mostly showcases how creating a network in Python code can be a lot more
    # flexible than a configuration file. Note that to make the code easier,
    # all the layers are just called `network` -- there is no need to give them
    # different names if all we return is the last one we created anyway; we
    # just used different names above for clarity.

    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)
    # Hidden layers and dropout:
    nonlin = lasagne.nonlinearities.rectify
    for _ in range(depth):
        network = lasagne.layers.DenseLayer(
                network, width, nonlinearity=nonlin)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)
    # Output layer:
    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(network, 10, nonlinearity=softmax)
    return network


def build_cnn(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 1, 50, BUFFER),
                                        input_var=input_var)


    network = lasagne.layers.Conv2DLayer(
        network, num_filters=16, filter_size=(2, 2),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(4, 2))

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=10, filter_size=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3))
    # lasagne.layers.get_output_shape(network)



    # # A fully-connected layer of 256 units with 50% dropout on its inputs:
    # network = lasagne.layers.DenseLayer(
    #         lasagne.layers.dropout(network, p=.5),
    #         num_units=256,
    #         nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=3,
            nonlinearity=lasagne.nonlinearities.softmax)

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

def main(model='cnn', num_epochs=50):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(BANK_TRAIN_BASE, BANK_TEST_BASE, BANK_LIST)
    # X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(TKK_TRAIN_BASE, TKK_TEST_BASE, TKK_LIST)

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if model == 'mlp':
        network = build_mlp(input_var)
    elif model.startswith('custom_mlp:'):
        depth, width, drop_in, drop_hid = model.split(':', 1)[1].split(',')
        network = build_custom_mlp(input_var, int(depth), int(width),
                                   float(drop_in), float(drop_hid))
    elif model == 'cnn':
        network = build_cnn(input_var)
    else:
        print("Unrecognized model type %r." % model)
        return

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
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            # print(batch)
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
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
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        main(**kwargs)