import numpy as np

# from keras import Sequential
# from keras.layers import Dense
# from keras.optimizers import sgd

import tensorflow as tf

import pdb

class NeuralKeras:
    def __init__(self, shape):
        '''Simplest possible 2-layer perceptron

        Arg:
            shape - 3-tuple: (nb_inputs, nb_hidden, nb_outputs)
        '''

        if len(shape) != 3:
            raise ValueError('This implementation supports only 2 layer NN')

        self.nb_inputs = shape[0]
        self.nb_hidden = shape[1]
        self.nb_outputs = shape[2]

        self.layer_hid = tf.keras.layers.Dense(
            shape[1], 
            input_shape=(shape[0], ), 
            activation='sigmoid')

        self.layer_out = tf.keras.layers.Dense(
            shape[2], 
            activation='sigmoid'
            )
        
        
        self.model = tf.keras.models.Sequential()
        self.model.add(self.layer_hid)
        self.model.add(self.layer_out)
        self.model.compile(tf.keras.optimizers.SGD(lr=0.3), "mse")

    def get_weights(self, layer):
        if layer == 0:
            return self.layer_hid.get_weights()[0]
        elif layer == 1:
            return self.layer_out.get_weights()[0]
        else:
            raise ValueError('Only layers 0 and 1 are supported')

    def get_biases(self, layer):
        if layer == 0:
            return np.reshape(self.layer_hid.get_weights()[1], (1, -1))
        elif layer == 1:
            return np.reshape(self.layer_out.get_weights()[1], (1, -1))
        else:
            raise ValueError('Only layers 0 and 1 are supported')
        
    def set_weights(self, layer, value):
        if layer == 0:
            res = self.layer_hid.get_weights()
            assert res[0].shape == value.shape
            res[0] = value
            self.layer_hid.set_weights(res)
        elif layer == 1:
            res = self.layer_out.get_weights()
            assert res[0].shape == value.shape
            res[0] = value
            self.layer_out.set_weights(res)
        else:
            raise ValueError('Only layers 0 and 1 are supported')

    def set_biases(self, layer, value):
        if layer == 0:
            res = self.layer_hid.get_weights()
            assert res[1].shape[0] == value.shape[1]
            res[1] = value.flatten()
            self.layer_hid.set_weights(res)
        elif layer == 1:
            res = self.layer_out.get_weights()
            assert res[1].shape[0] == value.shape[1]
            res[1] = value.flatten()
            self.layer_out.set_weights(res)
        else:
            raise ValueError('Only layers 0 and 1 are supported')


    def fun_sigmoid(self, x, deriv=False):
        if deriv:
            return np.multiply(self.fun_sigmoid(x), (1 - self.fun_sigmoid(x)))
        return 1 / (1 + np.exp(-x))

    def forward(self, data):
        assert isinstance(data, np.ndarray)
        assert data.ndim == 2
        assert len(data) > 0
        assert data.shape[1] == self.nb_inputs

        return self.model.predict(data, batch_size=len(data))

    def backward(self, data, labels):
        assert isinstance(data, np.ndarray)
        assert data.ndim == 2
        assert len(data) > 0
        assert data.shape[1] == self.nb_inputs

        assert isinstance(labels, np.ndarray)
        assert labels.ndim == 2
        assert len(labels) == len(data)
        assert labels.shape[1] == self.nb_outputs
        
        # forward hidden layer
        temp = np.dot(data, self.weights_hidden)
        inputs_hidden = np.add(temp, self.biases_hidden)
        outputs_hidden = self.fun_sigmoid(inputs_hidden)

        # forward output layer
        temp = np.dot(outputs_hidden, self.weights_output)
        inputs_output =  np.add(temp, self.biases_output)
        outputs_output = self.fun_sigmoid(inputs_output)

        temp = (outputs_output - labels)
        error_term_out = temp * self.fun_sigmoid(inputs_output, deriv=True)

        d_weights_output = np.dot(outputs_hidden.T, error_term_out)
        d_biases_output = error_term_out


        temp = np.dot(error_term_out, self.weights_output.T)
        error_term_hid = temp * self.fun_sigmoid(inputs_hidden, deriv=True)

        d_weights_hidden = np.dot(data.T, error_term_hid)
        d_biases_hidden = error_term_hid


        res_b = [d_biases_hidden, d_biases_output]
        res_w = [d_weights_hidden, d_weights_output] 

        # sum biases in case data was multi-row
        res_b[0] = np.sum(res_b[0], axis=0, keepdims=True)
        res_b[1] = np.sum(res_b[1], axis=0, keepdims=True)

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

        self.model.train_on_batch(data, labels)


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

        # This doesn't replicate results from other libraries
        self.model.fit(data, labels, batch_size=batch_size, nb_epoch=1)

    def evaluate(self, data):
        total_error = 0
        count = 0
        for x, y in data:
            out = self.forward(x)
            total_error += np.sum(np.square(out - y))
            if np.argmax(out) == np.argmax(y):
                count += 1

        return total_error, count