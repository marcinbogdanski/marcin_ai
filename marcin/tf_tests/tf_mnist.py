#!/usr/bin/env python

import tensorflow as tf
import sklearn
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import dxlib.feeders as dxfd

import pdb

def load_mnist(test=False):
    from sklearn.datasets import fetch_mldata

    # loads data into ~/scikit_learn_data/
    mnist = fetch_mldata('MNIST original')

    # Convert to float32, range 0..1; shape [70000, 784]
    data = mnist.data.astype(np.float32) / 255.0

    # One-hot encode labels, shape [70000, 10]
    eye = np.eye(10, dtype=np.int32)
    tar = mnist.target.astype(np.int32)
    targets = eye[tar]

    train = dxfd.Feeder(data[0:55000], targets[0:55000])
    valid = dxfd.Feeder(data[55000:65000], targets[55000:65000])
    test  = dxfd.Feeder(data[65000:70000], targets[65000:70000])

    return train, valid, test

def show_pics(feeder, start, num, cols=10):
    rows = np.ceil(num/cols)

    fig = plt.figure()
    for n in range(num):
        temp = feeder.features[start + n]
        label  = feeder.labels[start + n]
        data = temp.reshape([28, 28])
        a = fig.add_subplot(cols, rows, n+1)
        #a.set_title('label: ' + str(label))
        plt.imshow(data, cmap='gray')
    plt.show()


train, valid, test = load_mnist()

show_pics(train, start=0, num=25, cols=5)

exit(0)



ff = dxfd.Feeder()

pdb.set_trace()

exit(0)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

features = mnist.train.images
labels = mnist.train.labels

A = tf.constant([[1, 2], [3, 4]], name='A')
b = tf.placeholder(tf.int32, [2], name = 'B')  # scalar
C = tf.add(A, b, name='C')

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

dataset = tf.contrib.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))

iterator = dataset.make_initializable_iterator()




with tf.Session() as sess:

    sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                          labels_placeholder: labels})

    writer = tf.summary.FileWriter('logdir', sess.graph)
    writer.flush()

    feed_dict = {b: [0, 1]}
    res = sess.run(C, feed_dict=feed_dict)
