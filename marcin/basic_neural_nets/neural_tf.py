import numpy as np
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
        self.nb_inputs = layers[0]
        self.nb_hidden = layers[1]
        self.nb_outputs = layers[2]


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
            self.outputs_from_output = tf.sigmoid(self.inputs_to_output,
                name='output')

        #
        #   Backward pass
        #
        self.data_targets = tf.placeholder(tf.float32, [None, None],
                                           name='data_targets')
        # Learning rate
        self.eta = tf.placeholder(tf.float32, [])
        # Length of data input (for batch learning)
        self.data_in_len = tf.cast(tf.shape(self.data_in)[0], tf.float32)
        # Factor applying learning rate
        self.factor = tf.divide(-1 * self.eta, self.data_in_len)
        with tf.name_scope(scope_output):
            with tf.name_scope('output_grad'):
                temp = tf.subtract(self.outputs_from_output, self.data_targets)
                sig_der = tf.multiply(tf.sigmoid(self.inputs_to_output),
                    1 - tf.sigmoid(self.inputs_to_output))
                err_term_out = tf.multiply(temp, sig_der)
                self.delta_weights_output = \
                    tf.matmul(tf.transpose(self.output_from_hidden),
                    err_term_out, name='delta_weights_output');
                self.delta_biases_output = \
                    tf.reduce_sum(err_term_out, axis=0,
                        keep_dims=True, name='delta_biases_output')

                self.update_weights_output = \
                    tf.assign_add(self.weights_output, 
                        self.delta_weights_output * self.factor,
                        name='update_weights_output')
                self.update_biases_output = \
                    tf.assign_add(self.biases_output,
                        self.delta_biases_output * self.factor,
                        name='update_biases_output')

            with tf.name_scope('hidden_grad'):
                temp = tf.matmul(err_term_out, 
                    tf.transpose(self.weights_output))
                self.sig_der_hid = tf.multiply(tf.sigmoid(self.inputs_to_hidden),
                    1 - tf.sigmoid(self.inputs_to_hidden))
                err_term_hid = tf.multiply(temp, self.sig_der_hid)
                self.delta_weights_hidden = \
                    tf.matmul(tf.transpose(self.data_in),
                    err_term_hid, name='delta_weights_hidden')
                self.delta_biases_hidden = \
                    tf.reduce_sum(err_term_hid, axis=0,
                        keep_dims=True, name='delta_biases_hidden')

                self.update_weights_hidden = \
                    tf.assign_add(self.weights_hidden, 
                        self.delta_weights_hidden * self.factor,
                        name='update_weights_hidden')
                self.update_biases_hidden = \
                    tf.assign_add(self.biases_hidden,
                        self.delta_biases_hidden * self.factor,
                        name='update_biases_hidden')




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

        return out[0]


    def forward(self, data):
        assert isinstance(data, np.ndarray)
        assert data.ndim == 2
        assert len(data) > 0
        assert data.shape[1] == self.nb_inputs

        return self.sess.run(self.outputs_from_output,
                             feed_dict={self.data_in: data})

    def backward(self, data, labels, eta=0.0, apply_deltas=True):
        assert isinstance(data, np.ndarray)
        assert data.ndim == 2
        assert len(data) > 0
        assert data.shape[1] == self.nb_inputs

        assert isinstance(labels, np.ndarray)
        assert labels.ndim == 2
        assert len(labels) == len(data)
        assert labels.shape[1] == self.nb_outputs

        assert isinstance(eta, float)
        assert isinstance(apply_deltas, bool)

        lenn = self.sess.run(self.data_in_len, 
            feed_dict={self.data_in: data, self.data_targets: labels})

        if apply_deltas == False:
            # Just run backprop, do not apply delta-weight
            delta_weights_output, delta_biases_output, \
            delta_weights_hidden, delta_biases_hidden = self.sess.run(
                [self.delta_weights_output, self.delta_biases_output,
                self.delta_weights_hidden, self.delta_biases_hidden],
                feed_dict={self.data_in: data, self.data_targets: labels});
        else:
            delta_weights_output, delta_biases_output, \
            delta_weights_hidden, delta_biases_hidden, \
            _, _, _, _ = self.sess.run(
                [self.delta_weights_output, self.delta_biases_output,
                self.delta_weights_hidden, self.delta_biases_hidden,
                self.update_weights_output, self.update_biases_output,
                self.update_weights_hidden, self.update_biases_hidden],
                feed_dict={self.data_in: data,
                           self.data_targets: labels,
                           self.eta: eta});

        res_w = [delta_weights_hidden, delta_weights_output]
        res_b = [delta_biases_hidden, delta_biases_output]

        return res_b, res_w

    def train_batch(self, data, labels, eta):
        assert isinstance(data, np.ndarray)
        assert data.ndim == 2
        assert len(data) > 0
        assert data.shape[1] == self.nb_inputs

        assert isinstance(labels, np.ndarray)
        assert labels.ndim == 2
        assert len(labels) == len(data)
        assert labels.shape[1] == self.nb_outputs

        assert isinstance(eta, float)
        assert eta >= 0

        del_b, del_w = self.backward(data, labels, eta)

        return del_b, del_w


    def train_SGD(self, data, labels, batch_size, eta, callback=None):
        assert isinstance(data, np.ndarray)
        assert data.ndim == 2
        assert len(data) > 0
        assert data.shape[1] == self.nb_inputs

        assert isinstance(labels, np.ndarray)
        assert labels.ndim == 2
        assert len(labels) == len(data)
        assert labels.shape[1] == self.nb_outputs

        assert isinstance(batch_size, int)
        assert batch_size > 0
        assert isinstance(eta, float)
        assert eta >= 0

        indices = np.array(range(len(data)))
        np.random.shuffle(indices)

        for k in range(0, len(indices), batch_size):
            idx = indices[k:k + batch_size]
            del_b, del_w = self.train_batch(data[idx], labels[idx], eta)

            if callback is not None:
                callback(self)

    def evaluate(self, data):
        total_error = 0
        count = 0
        for x, y in data:
            out = self.forward(x)
            total_error += np.sum(np.square(out - y))
            if np.argmax(out) == np.argmax(y):
                count += 1

        return total_error, count