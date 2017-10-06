import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pdb

def pd(string, stuff):
    print(string, stuff, type(stuff))

def reset_default_graph():
    tf.reset_default_graph()

class NeuralNetworkTF(object):

    def __init__(self, layers):
        '''Simplest possible 2-layer perceptron

        Arg:
            shape - 3-tuple: (nb_inputs, nb_hidden, nb_outputs)
        '''


        # Input to sigmoid function 
        # (this is just for fun_simoid test,
        # which is not otherwise used)
        with tf.variable_scope('test_sigmoid'):
            self.sig_in = tf.placeholder(tf.float32, [None], name='sig_in')
            self.sig_out = tf.sigmoid(self.sig_in, name='sig_out')
            self.sig_deriv = tf.multiply(tf.sigmoid(self.sig_in),
                1 - tf.sigmoid(self.sig_in))

        #
        #   Variables for hidden layer
        #
        with tf.name_scope('hidden_layer') as scope_hidden:
            # Weights
            self.weights_hidden = tf.Variable(tf.truncated_normal(
                shape=[layers[0], layers[1]],
                mean=0, stddev=1/np.sqrt(layers[0])), name='weights_hidden')
            
            self.biases_hidden = tf.Variable(tf.truncated_normal(
                shape=[1, layers[1]], mean=0, stddev=1), name='biases_hidden')
        

        #
        #   Variables for output layer
        #
        with tf.name_scope('output_layer') as scope_output:
            self.weights_output = tf.Variable(tf.truncated_normal(
                shape=[layers[1], layers[2]],
                mean=0, stddev=1/np.sqrt(layers[1])), name='weights_output')

            self.biases_output = tf.Variable(tf.truncated_normal(
                shape=[1, layers[2]], mean=0, stddev=1), name='biases_output')

        #   Variable setters
        with tf.name_scope('variable_setters'):
            # For setting weights and biases (ok to re-use one placeholder?)
            self.temp = tf.placeholder(tf.float32, [None, None], name='temp')
            self.assigh_wh = tf.assign(self.weights_hidden, self.temp, name='assign_wh')
            self.assigh_bh = tf.assign(self.biases_hidden, self.temp, name='assign_bh')
            self.assigh_wo = tf.assign(self.weights_output, self.temp, name='assign_wo')
            self.assigh_bo = tf.assign(self.biases_output, self.temp, name='assign_bo')

        #
        #   Forward pass
        #
        self.data_in = tf.placeholder(tf.float32, [None, None], name='data_in')

        with tf.name_scope(scope_hidden):
            temp = tf.matmul(self.data_in, self.weights_hidden)
            self.inputs_to_hidden = tf.add(temp, self.biases_hidden)
            self.output_from_hidden = tf.sigmoid(self.inputs_to_hidden)

        with tf.name_scope(scope_output):
            temp = tf.matmul(self.output_from_hidden, self.weights_output)
            self.inputs_to_output = tf.add(temp, self.biases_output)
            self.outputs_from_output = tf.sigmoid(self.inputs_to_output)

        #
        #   Backward pass
        #
        self.data_targets = tf.placeholder(tf.float32, [None, None], name='data_targets')
        with tf.name_scope(scope_output):
            temp = tf.subtract(self.outputs_from_output, self.data_targets)
            sig_der = tf.multiply(tf.sigmoid(self.inputs_to_hidden),
                1 - tf.sigmoid(self.inputs_to_hidden))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.writer = tf.summary.FileWriter('logdir', self.sess.graph)
        self.writer.flush()  # necessary?

    def close_tf_session(self):
        self.sess.close()

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
        
    def set_weights(self, layer, value):
        if layer == 0:
            self.sess.run(self.assigh_wh, feed_dict={self.temp: value})
        elif layer == 1:
            self.sess.run(self.assigh_wo, feed_dict={self.temp: value})
        else:
            raise ValueError('Only layers 0 and 1 are supported')

    def set_biases(self, layer, value):
        if layer == 0:
            self.sess.run(self.assigh_bh, feed_dict={self.temp: value})
        elif layer == 1:
            self.sess.run(self.assigh_bo, feed_dict={self.temp: value})
        else:
            raise ValueError('Only layers 0 and 1 are supported')

    def fun_sigmoid(self, x, deriv=False):
        x = np.array([x])
        if deriv == True:
            out = self.sess.run(self.sig_deriv, feed_dict={self.sig_in: x})
        else:
            out = self.sess.run(self.sig_out, feed_dict={self.sig_in: x})

        print(out)
        return out[0]


    def forward(self, data):
        return self.sess.run(self.outputs_from_output,
                             feed_dict={self.data_in: data})