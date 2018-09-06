#!/usr/bin/env python

import tensorflow as tf
import sklearn
import time
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

    # Shuffle data
    perm0 = np.arange(len(data))
    np.random.shuffle(perm0)
    data = data[perm0]
    targets = targets[perm0]

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


# Explore data
train, valid, test = load_mnist()
# show_pics(train, start=0, num=25, cols=5)


batches = 1000
batch_size = 100
learning_rate = 0.5


data = tf.placeholder(tf.float32, [None, 784])
labels = tf.placeholder(tf.float32, [None, 10])

weights = tf.get_variable('weights', [784, 10],
    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0/784))
biases = tf.get_variable('biases', [10],
    initializer=tf.zeros_initializer())
logits = tf.add(tf.matmul(data, weights), biases, name='logits')



cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
   labels=labels, logits=logits)
loss = tf.reduce_mean(cross_entropy)


optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

is_correct = \
    tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1),
        name='is_correct')
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32), name='accuracy')



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('logdir', sess.graph)
    writer.flush()

    while True:
        if train.batches_completed >= batches:
            break

        in_data, in_labels = train.next_batch(batch_size)
        acc, ll, _ = sess.run([accuracy, loss, optimizer],
            feed_dict={data: in_data,
                        labels: in_labels})

        if train.batches_completed % 100 == 0:
            valid_acc, valid_loss = sess.run([accuracy, loss],
                feed_dict={data: valid.features,
                           labels: valid.labels})
            
            print('epoch: ', train.epochs_completed, '   ',
                  'batch:', train.batches_completed, '   ',
                  'valid_acc', valid_acc)


    
    acc, ll, _ = sess.run([accuracy, loss, optimizer],
            feed_dict={data: test.features,
                        labels: test.labels})
    print('Final accuracy:', acc)


    