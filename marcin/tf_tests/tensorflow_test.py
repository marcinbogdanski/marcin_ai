#!/usr/bin/env python

import tensorflow as tf
import sys

import pdb

A = tf.constant([[1, 2], [3, 4]], name='A')
b = tf.placeholder(tf.int32, [2], name = 'B')  # scalar
C = tf.add(A, b, name='E')


# graph_def = tf.get_default_graph().as_graph_def()
# graphpb_txt = str(a.graph.as_graph_def())
# with open('graphpb.txt', 'w') as f: f.write(graphpb_txt)





with tf.Session() as sess:

    writer = tf.summary.FileWriter('logdir', sess.graph)
    #writer.flush()

    feed_dict = {b: [0, 1]}
    res = sess.run(C, feed_dict=feed_dict)
