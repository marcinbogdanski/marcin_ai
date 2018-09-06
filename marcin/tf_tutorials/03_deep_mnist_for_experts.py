'''
Following this tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/beginners

Objective is to classify MNIST dataset using simple one layer neural net
'''

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import pickle

import sys, os, importlib
sys.path.append(os.getcwd())


def load_data(filepath):
    '''Loads MNIST dataset from HDD
    
    Params:
        filepath - path to  mnist.pkl.gz file
        
    Returns:
        train_features - shape [60000, 786]
        train_labels - one hot encoded, shape [60000, 10]
        test_features - shape [10000, 786]
        test_labels - one hot encoded, shape [10000, 10]
    '''
    import pickle, gzip
    
    with gzip.open(filepath, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        train_set, valid_set, test_set = u.load()
        
    train_x, train_l = train_set[0], train_set[1]
    valid_x, valid_l = valid_set[0], valid_set[1]
    test_x, test_l = test_set[0], test_set[1]
    
    train_x = np.concatenate([train_x, valid_x])
    train_l = np.concatenate([train_l, valid_l])
    
    # convert to one hot
    train_y = np.zeros( [len(train_l), 10] )
    train_y[ np.arange(len(train_l)), train_l ] = 1
   
    test_y = np.zeros( [len(test_l), 10] )
    test_y[ np.arange(len(test_l)), test_l ] = 1
    
    return train_x, train_y, test_x, test_y

def main():
        
    # define inputs
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
   
    # define model
    # single layer NN with 10 neurons
    W = tf.Variable( tf.zeros( [784, 10] ) )
    b = tf.Variable( tf.zeros( [10] ) )
    y = tf.nn.softmax( tf.matmul( x, W ) + b )

    # loss is cross entropy function
    loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    # learning rate = 0.5
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    # check if feature is correctly classified
    is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # ratio of correctly classified examples
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    
    # load data
    train_x, train_y, test_x, test_y = load_data('..\mnist\mnist.pkl.gz')
            
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
                
        counter = 0
        while counter < 1000:
            # shuffle whole training set
            idx = np.arange(len(train_x))
            np.random.shuffle(idx)
            tx = train_x[idx]
            ty = train_y[idx]
                        
            # iterate in small batches
            batch_size = 100
            for j in range(0, len(tx), batch_size):
                feed = { x: tx[j:j+batch_size], y_: ty[j:j+batch_size] }
                sess.run(train_step, feed)
                counter += 1
                                   
        
        feed = { x: test_x, y_: test_y}
        res = sess.run(accuracy, feed)
        print('accuracy:', round(res * 100), '%')
                    
        
    
    
print('hoho')
print(__name__)
    
if __name__ == '__main__':
    main()
    print('after main')