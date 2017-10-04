#!/usr/bin/env python

import tensorflow as tf
import sklearn
import sys, os

#parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#os.sys.path.insert(0,parentdir) 


import pdb

import dxlib.feeders as dxfd

ff = dxfd.Feeder()

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
