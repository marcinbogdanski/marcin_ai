from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt

def pd(string, stuff):
    print(string, stuff, type(stuff))

class NeuralNetworkTF(object):
    """Basic multi-layer neural network class.
    
    Attributes:
        layers (list): Number of neuros in each layer, starting from 1st hidden
        num_inputs (int): Number of inputs to neural netowrk
        num_outputs (int): Number of ouputs from neural netowrk
        num_layers (int): Number of layers in neural network (hidden + output)
        shapes (list): list of shapes for weight
        
        weights[layer][n][m]: python list of 2d np.array
            layer: [layer=0] 1st_hidden; [layer=1] 2nd_hidden etc.
            n: number of neurons in a layer(rows)
            m: number of inputs from input or previous layer (columns)
    """

    def __init__(self, layers):
        """
        Params:
            inputs (list): number of neurons in particular layers
                     e.g. [4, 8, 1] means:
                     4x inputs (not neurons)
                     8x neurons in first hidden layer
                     1x neuron in ouput layer
        """
        # just a placeholders
        self.layers = layers[1:]
        self.num_inputs = layers[0]
        self.num_outputs = layers[-1]
        self.num_layers = len(layers) - 1
        
        self.bias_mult = 1    # set to zero to disable bias
        
        self.init_sqrt(layers)
        self.activations = [self.fun_sigmoid] * self.num_layers
        
        #self.init_relu(layers)
        #self.activations = [self.fun_relu] * self.num_layers

    def init_norm(self, layers):
        self.shapes = []
        self.weights = []
        for i in range(1, len(layers)):
            num_neurons = layers[i]
            num_inputs = layers[i - 1]
            self.shapes.append( (num_inputs, num_neurons) )
            self.weights.append( np.random.randn(num_inputs, num_neurons) )
                   
        self.biases = [np.random.randn(1, n) for n in layers[1:]]
        #self.biases = [np.zeros((1,n)) for n in layers[1:]]
        self.inputs = [np.zeros((1, n)) for n in layers[1:]]
        self.outputs = [np.zeros((1, n)) for n in layers[1:]]
        self.errors = [np.zeros((1, n)) for n in layers[1:]]
        
    def init_sqrt(self, layers):
        self.init_norm(layers)
        
        #init first layer
        shape = self.weights[0].shape
        self.weights[0] = np.random.normal(0, 1 / np.sqrt(self.num_inputs), shape)
        assert( shape == self.weights[0].shape )
        
        for l in range(1, self.num_layers):
            shape = self.weights[l].shape
            self.weights[l] = np.random.normal(0, 1 / np.sqrt(self.layers[l - 1]), shape)
            assert( shape == self.weights[l].shape )
            
    def init_relu(self, layers):
        self.init_norm(layers)
        
        #init first layer
        shape = self.weights[0].shape
        self.weights[0] = np.random.normal(0, 2 / np.sqrt(self.num_inputs), shape)
        assert( shape == self.weights[0].shape )
        
        for l in range(1, self.num_layers):
            shape = self.weights[l].shape
            self.weights[l] = np.random.normal(0, 2 / np.sqrt(self.layers[l - 1]), shape)
            assert( shape == self.weights[l].shape )
                
    def fun_sigmoid(self, x, deriv=False):
        if deriv:
            return np.multiply(self.fun_sigmoid(x), (1 - self.fun_sigmoid(x)))
        return 1 / (1 + np.exp(-x))
        
    def fun_linear(self, x, deriv=False):
        if deriv:
            return 1
        return x
    
    def fun_relu(self, x, deriv=False):
        if deriv:
            1. * (x >= 0)
        return x * (x >= 0)
     
    def _check_fix_input(self, data, expected_length):
        """
            Check shape and convert to row vector if necessary.
        """       
        if np.isscalar(data):
            print('Check: isscalar')
            raise
            data = np.asarray([[data]])  # wrap scalars
        if isinstance(data, (tuple, list)):
            print('Check: tuple, list')
            raise
            data = np.array( [data] )  # we work with row vectors here
        if len(data.shape) == 1:
            #print('Check: 1D')
            #raise
            #data = np.array( [data] ) # convert (n,) 1D shape to 2D shape
            pass
        if isinstance(data, np.matrix):
            print('Check: matrix')
            raise
            data = np.asarray(data)  # convert matrix to array

        assert( data.shape == (2,) or
                data.shape[1] == expected_length )
        return data
        
    def _check_weight_shapes(self):
        for i in range(self.num_layers):
            # Make sure we didn't mess up something when multiplying matrices
            assert( self.weights[i].shape == self.shapes[i] )
    

    def forward(self, data):
        """
        Params:
            data - input data, each row is one feature set,
                   multiple rows will forward network on whole data set in one go
        Returns:
            nn output - row format
        """
        print('hoho')

        data = self._check_fix_input(data, self.num_inputs)
        
        # Calculate first layer output
        self.inputs[0] = np.dot(data, self.weights[0]) + self.biases[0] * self.bias_mult
        self.outputs[0] = self.activations[0](self.inputs[0])
        
        # Calculate other hidden and output layers
        for n in range(1, self.num_layers):
            self.inputs[n] = np.dot(self.outputs[n - 1], self.weights[n]) \
                + self.biases[n] * self.bias_mult
            self.outputs[n] = self.activations[n](self.inputs[n])
            
        self._check_weight_shapes()
            
        # Convert to row vector
        return self.outputs[-1]

    def backward(self, data, target):
        """
        Peform single fwd and backward pass
        Returns:
            array containing delta-changes for all weights in all layers
        """
        data = self._check_fix_input(data, self.num_inputs)
        target = self._check_fix_input(target, self.num_outputs)
        
        if len(data.shape) == 1:
            data = np.array( [data] )  # convert (n,) 1D shape to 2D shape
        
        self.forward(data)
        
        delta_weights = [np.zeros((w.shape)) for w in self.weights]
        delta_biases = [None for b in self.biases]
        
        for l in reversed(range(self.num_layers)):
          
            # Calculate error
            if l == self.num_layers - 1:
                error = self.outputs[-1] - target  # Output layer
            else:
                error = np.dot(self.errors[l + 1], self.weights[l + 1].T)  # Hidden
            
            deriv = self.activations[l](self.inputs[l], True)
            self.errors[l] = np.multiply(deriv, error)
            
            # Calculate delta weight arrays
            if l == 0:
                delta_weights[l] = np.dot(data.T, self.errors[l])
                delta_biases[l] = self.errors[l]
            else:
                delta_weights[l] = np.dot(self.outputs[l - 1].T, self.errors[l])
                delta_biases[l] = self.errors[l]
        
            assert delta_weights[l].shape == self.shapes[l]
            
        return delta_biases, delta_weights
     
    def train_batch(self, batch, eta, lmbda=0.0, n=1):
        if(len(batch)) == 0:
            return
        
        inputs = batch[0][0]
        targets = batch[0][1]
        del_b, del_w = self.backward(inputs, targets)
        
        for i in range(1, len(batch)):
            inputs = batch[i][0]
            targets = batch[i][1]
            temp_del_b, temp_del_w = self.backward(inputs, targets)
            del_b = [db + tdb for db, tdb in zip(del_b, temp_del_b)]
            del_w = [dw + tdw for dw, tdw in zip(del_w, temp_del_w)]
            
        for i in range(self.num_layers):
            self.weights[i] = (1 - eta * (lmbda / n)) * self.weights[i] - (eta / len(batch)) * del_w[i]
            self.biases[i] += -eta / len(batch) * del_b[i]
    
    def train_SGD(self, data, batch_size, eta, callback=None):
        temp_shuffled = data[:]
        np.random.shuffle(temp_shuffled)

        for k in range(0, len(temp_shuffled), batch_size):
            data_batch = temp_shuffled[k:k + batch_size]
            self.train_batch(data_batch, eta)
            
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
        
    def evaluate2(self, data, targets):
        out = self.forward(data)
        return np.mean((out - targets)**2), 0
        return np.sum(np.square(out - targets)), 0
        
        
    def print_debug(self):
        for n in range(self.num_layers):
            print('Layer: ', n, 'neurons:', self.layers[n])
            print('Shapes: ', self.shapes)
            print('Weights: ', type(self.weights[n]))
            print(self.weights[n])
            print('Biases: ', type(self.biases[n]))
            print(self.biases[n])
            print('Inputs: ', type(self.inputs[n]))
            print(self.inputs[n])
            print('Outputs: ', type(self.outputs[n]))
            print(self.outputs[n])
            print()
            
    def vis_2D(self, layer_nb, neuron_nb, first=False):
        px = np.linspace(0, 1, 50)
        py = np.linspace(0, 1, 50)
        pz = np.zeros((len(px), len(px)))
        
        for i in range(len(px)):
            for j in range(len(py)):
                self.forward( (px[i], py[j]) )
                pz[i, j] = self.outputs[layer_nb][neuron_nb][0]
       
        if first:
            plt.figure()
        else:
            plt.clf()

        extent = [px[0], px[-1], py[0], py[-1]]
        plt.imshow(pz.T, extent=extent, origin='lower')

        if first:
            plt.colorbar()
        
        #plt.show()
        plt.pause(0.01)

