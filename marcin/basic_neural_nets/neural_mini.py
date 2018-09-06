import numpy as np

class NeuralNetwork2:
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
        
        self.weights = [None, None]
        self.biases = [None, None]

        # hidden layer
        self.weights_hidden = np.random.randn(shape[0], shape[1])
        self.biases_hidden = np.random.randn(1, shape[1])

        # output layer
        self.weights_output = np.random.randn(shape[1], shape[2])
        self.biases_output = np.random.randn(1, shape[2])

    def get_weights(self, layer):
        if layer == 0:
            return self.weights_hidden
        elif layer == 1:
            return self.weights_output
        else:
            raise ValueError('Only layers 0 and 1 are supported')

    def get_biases(self, layer):
        if layer == 0:
            return self.biases_hidden
        elif layer == 1:
            return self.biases_output
        else:
            raise ValueError('Only layers 0 and 1 are supported')
        
    def set_weights(self, layer, value):
        if layer == 0:
            assert self.weights_hidden.shape == value.shape
            self.weights_hidden = value
        elif layer == 1:
            assert self.weights_output.shape == value.shape
            self.weights_output = value
        else:
            raise ValueError('Only layers 0 and 1 are supported')

    def set_biases(self, layer, value):
        if layer == 0:
            assert self.biases_hidden.shape == value.shape
            self.biases_hidden = value
        elif layer == 1:
            assert self.biases_output.shape == value.shape
            self.biases_output = value
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

        # hidden layer
        temp = np.dot(data, self.weights_hidden)
        inputs_hidden = np.add(temp, self.biases_hidden)
        outputs_hidden = self.fun_sigmoid(inputs_hidden)

        # output layer
        temp = np.dot(outputs_hidden, self.weights_output)
        inputs_output =  np.add(temp, self.biases_output)
        outputs_output = self.fun_sigmoid(inputs_output)

        return outputs_output

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

        del_b, del_w = self.backward(data, labels)

        self.weights_hidden += -eta / len(data) * del_w[0]
        self.weights_output += -eta / len(data) * del_w[1]
        self.biases_hidden += -eta / len(data) * del_b[0]
        self.biases_output += -eta / len(data) * del_b[1]


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
            self.train_batch(data[idx], labels[idx], eta)

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