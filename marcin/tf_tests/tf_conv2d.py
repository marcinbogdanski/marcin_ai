#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import pdb



inputt = tf.placeholder(tf.float32, [1, 4, 4, 1])

dataA = np.array([[[[0.0], [1.0], [0.0], [0.0]],
                   [[1.0], [1.0], [1.0], [1.0]],
                   [[0.0], [1.0], [0.0], [1.0]],
                   [[0.0], [1.0], [1.0], [1.0]]]])

filt1 = np.array([[[-1], [ 1], [-1]],   # detects crosses
                  [[ 1], [ 1], [ 1]],
                  [[-1], [ 1], [-1]]])
filt2 = np.array([[[ 1], [ 1], [ 1]],   # detects circles
                  [[ 1], [-1], [ 1]],
                  [[ 1], [ 1], [ 1]]])
ff = np.stack([filt1, filt2], axis=-1)


# plt.figure()
# plt.imshow(dataA)
# plt.colorbar()
# plt.show()
# exit()
const = tf.constant(ff, dtype=tf.float32)
weights = tf.get_variable('weights', initializer=const)
filt = tf.nn.conv2d(
    inputt, weights, strides=[1, 1, 1, 1], padding='SAME')

maxx = tf.nn.max_pool(
    filt, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('logdir', sess.graph)
    writer.flush()

    # out = [batch_size, height, width, k_size]
    out_filt, out_max = sess.run([filt, maxx], feed_dict={inputt: dataA})
    
                            # out_filt = [batch_size, height, width, k_size]
    example = out_filt[0]   # example = [heigth, width, k_size]
    res1, _ = np.dsplit(example, 2)  # res1 = [height, width, 1]
    res1 = res1.squeeze()            # res1 = [height, width]

    plt.imshow(res1)
    plt.colorbar()
    plt.show()


    example = out_max[0]
    max1, _ = np.dsplit(example, 2)
    max1 = max1.squeeze()

    plt.imshow(max1)
    plt.colorbar()
    plt.show()

    #pdb.set_trace()

    print('hoho')
    
    # acc, ll, _ = sess.run([accuracy, loss, optimizer],
    #         feed_dict={data: test.features,
    #                     labels: test.labels})
    # print('Final accuracy:', acc)


