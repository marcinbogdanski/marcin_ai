import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pdb

def pd(string, stuff):
    print(string, stuff, type(stuff))

class NeuralNetworkTF(object):

    def __init__(self, layers):
        '''Simplest possible 2-layer perceptron

        Arg:
            shape - 3-tuple: (nb_inputs, nb_hidden, nb_outputs)
        '''


        # Input to sigmoid function 
        # (this is just for fun_simoid test,
        # which is not otherwise used)
        self.sig_in = tf.placeholder(tf.float32, [None], name='sig_in')
        self.sig_out = tf.sigmoid(self.sig_in, name='sig_out')

        # Weights
        self.weights_hidden = tf.Variable(tf.truncated_normal(
            shape=[layers[0], layers[1]],
            mean=0, stddev=1/np.sqrt(layers[0])), name='weights_hidden')

        self.weights_output = tf.Variable(tf.truncated_normal(
            shape=[layers[1], layers[2]],
            mean=0, stddev=1/np.sqrt(layers[1])), name='weights_output')

        self.biases_hidden = tf.Variable(tf.truncated_normal(
            shape=[1, layers[1]], mean=0, stddev=1), name='biases_hidden')
        self.biases_output = tf.Variable(tf.truncated_normal(
            shape=[1, layers[2]], mean=0, stddev=1), name='biases_output')


        # For setting weights and biases (ok to re-use one placeholder?)
        self.temp = tf.placeholder(tf.float32, [None, None], name='temp')
        self.assigh_wh = tf.assign(self.weights_hidden, self.temp, name='assign_wh')
        self.assigh_bh = tf.assign(self.biases_hidden, self.temp, name='assign_bh')
        self.assigh_wo = tf.assign(self.weights_output, self.temp, name='assign_wo')
        self.assigh_bo = tf.assign(self.biases_output, self.temp, name='assign_bo')

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_weights(self, layer):
        if layer == 0:
            return self.sess.run(self.weights_hidden)
        elif layer == 1:
            return self.sess.run(self.weights_output)
        else:
            raise ValueError('Only layers 0 and 1 are supported')

    def get_biases(self, layer):
        if layer == 0:
            return self.sess.run(self.biases_hidden)
        elif layer == 1:
            return self.sess.run(self.biases_output)
        else:
            raise ValueError('Only layers 0 and 1 are supported')
        
    def set_weights(self, layer, new_value):
        if layer == 0:
            self.sess.run(self.assigh_wh, feed_dict={self.temp: new_value})
        elif layer == 1:
            self.sess.run(self.assigh_wo, feed_dict={self.temp: new_value})
        else:
            raise ValueError('Only layers 0 and 1 are supported')

    def set_biases(self, layer, new_value):
        if layer == 0:
            self.sess.run(self.assigh_bh, feed_dict={self.temp: new_value})
        elif layer == 1:
            self.sess.run(self.assigh_bo, feed_dict={self.temp: new_value})
        else:
            raise ValueError('Only layers 0 and 1 are supported')

    def fun_sigmoid(self, x, deriv=False):
        x = np.array([x])
        out = self.sess.run(self.sig_out, feed_dict={self.sig_in: x})
        return out