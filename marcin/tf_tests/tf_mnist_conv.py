#!/usr/bin/env python

import tensorflow as tf
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
    data = data.reshape([len(data), 28, 28, 1])

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
    rows = np.ceil(num / cols)

    fig = plt.figure()
    for n in range(num):
        temp = feeder.features[start + n]
        label  = feeder.labels[start + n]
        data = temp.reshape([28, 28])
        a = fig.add_subplot(cols, rows, n + 1)
        a.set_title('label: ' + str(label))
        plt.imshow(data, cmap='gray')
    plt.show()


# Explore data
train, valid, test = load_mnist()
# show_pics(train, start=0, num=25, cols=5)


epochs = 4
batch_size = 128
learning_rate = 0.001
dropout_keep_prob = 0.75
test_valid_size = 256


inputs = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input')
labels = tf.placeholder(tf.float32, [None, 10], name='labels')
keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')

with tf.variable_scope('conv_14x14x32'):
    w_1 = tf.get_variable('w_1', [5, 5, 1, 32],
        initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
    b_1 = tf.get_variable('b_1', [32],
        initializer=tf.constant_initializer(0.1))
    l1_conv2d = tf.nn.conv2d(inputs, w_1, strides=[1, 1, 1, 1], padding='SAME')
    l1_bias = tf.nn.bias_add(l1_conv2d, b_1)
    l1_relu = tf.nn.relu(l1_bias)
    l1_maxpool = tf.nn.max_pool(
        l1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.variable_scope('conv_7x7x64'):
    w_2 = tf.get_variable('w_2', [5, 5, 32, 64],
        initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
    b_2 = tf.get_variable('b_2', [64],
        initializer=tf.constant_initializer(0.1))
    l2_conv2d = tf.nn.conv2d(l1_maxpool, w_2, strides=[1, 1, 1, 1], padding='SAME')
    l2_bias = tf.nn.bias_add(l2_conv2d, b_2)
    l2_relu = tf.nn.relu(l2_bias)
    l2_maxpool = tf.nn.max_pool(
        l2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.variable_scope('hidden_1024'):
    w_3 = tf.get_variable('w_3', [7*7*64, 1024],
        initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
    b_3 = tf.get_variable('b_3', [1024],
        initializer=tf.constant_initializer(0.1))
    l3_reshape = tf.reshape(l2_maxpool, [-1, 7*7*64])
    l3_logits = tf.add(tf.matmul(l3_reshape, w_3), b_3)
    l3_relu = tf.nn.relu(l3_logits)
    l3_dropout = tf.nn.dropout(l3_relu, keep_prob)

with tf.variable_scope('fully_connected'):
    w_4 = tf.get_variable('w_4', [1024, 10],
        initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
    b_4 = tf.get_variable('b_4', [10],
        initializer=tf.constant_initializer(0.1))
    l4_logits = tf.add(tf.matmul(l3_dropout, w_4), b_4)

with tf.variable_scope('ce_loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
       labels=labels, logits=l4_logits)
    loss = tf.reduce_mean(cross_entropy)
    

with tf.variable_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.variable_scope('evaluation'):
    is_correct = \
        tf.equal(tf.argmax(l4_logits, 1), tf.argmax(labels, 1),
            name='is_correct')
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32), name='accuracy')
    

with tf.name_scope('summaries'):
    tf.summary.image('inputs', inputs)
    tf.summary.histogram('w_1', w_1)
    tf.summary.histogram('b_1', b_1)
    tf.summary.histogram('w_2', w_2)
    tf.summary.histogram('b_2', b_2)
    tf.summary.histogram('w_3', w_3)
    tf.summary.histogram('b_3', b_3)
    tf.summary.histogram('w_4', w_4)
    tf.summary.histogram('b_4', b_4)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logdir', sess.graph)
    writer.flush()

    while True:
        it = train.batches_completed
        if train.epochs_completed >= epochs:
            break

        batch_inputs, batch_labels = train.next_batch(batch_size)
        
        if it == 0 or it % 5 == 0:
            s = sess.run(merged_summary, 
                feed_dict={inputs: batch_inputs,
                        labels: batch_labels,
                        keep_prob: dropout_keep_prob})
            writer.add_summary(s, train.batches_completed)

        acc, ll, _ = sess.run([accuracy, loss, optimizer],
            feed_dict={inputs: batch_inputs,
                        labels: batch_labels,
                        keep_prob: dropout_keep_prob})


        if it == 0 or it % 5 == 0:
            batch_inputs, batch_labels = valid.next_batch(test_valid_size)
            valid_acc, valid_loss = sess.run([accuracy, loss],
                feed_dict={inputs: batch_inputs,
                           labels: batch_labels,
                           keep_prob: 1.})
            
            print('epoch: ', train.epochs_completed, '   ',
                  'batch:', train.batches_completed, '   ',
                  'valid_acc', valid_acc)


    batch_inputs, batch_labels = test.next_batch(test_valid_size)
    acc, ll, _ = sess.run([accuracy, loss, optimizer],
            feed_dict={inputs: batch_inputs,
                        labels: batch_labels,
                        keep_prob: 1.})
    print('Final accuracy:', acc)


