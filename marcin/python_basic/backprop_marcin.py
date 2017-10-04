from __future__ import absolute_import, division, print_function, unicode_literals

import random
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, deriv=False):
    if not deriv:
        # sigmoid function
        return 1 / (1 + np.exp(-x))
    else:
        # first derivative of sigmoid
        return sigmoid(x) * (1 - sigmoid(x))
        
def relu(x, deriv=False):
    if not deriv:
        return x * (x>0)
    else:
        return np.array(x>0, dtype=x.dtype)
        
def tanh(x, deriv=False):
    if not deriv:
        # tanh function
        return np.tanh(x)
    else:
        # first derivative
        return 1 - (np.tanh(x) * np.tanh(x))
        
        
        
def vis2D(weights, biases, first=False):
    fsdf = fdasfas
    px = np.linspace(0, 1, 50)
    py = np.linspace(0, 1, 50)
    #vx, vy = np.meshgrid(px, py)
    #vx = vx.flatten()
    #vy = vy.flatten()
    pz = np.zeros((len(px), len(px)))
    
    for i in range(len(px)):
        for j in range(len(py)):
            X = np.array( (px[i], py[j]) )
            pz[i, j] = tanh(np.dot( X, weights) + biases)
            
    #print('px', px)
    #print('py', py)
    #print('pz', pz)
    
    # Generate some test data
    x = np.random.randn(100)
    y = np.random.randn(100)
    
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

class Network:
    def __init__(self, sizes, funct=sigmoid):
        '''params:
            sizes = (2, 2, 1) => 2 inputs, 2 hidden neurons, 1 output
            funct - neuron activation function
        '''
        self.funct = funct
        self.outputs = []
        
        for size in sizes:
            self.outputs.append( np.zeros( size ) )
            
        self.inputs = []    #  inputs[layer_nb][neuron_nb]
        self.weights = []   #  weights[layer_nb][neuron_nb][previous_layer_neuron_nb]
        self.biases = []    #  biases[layer_nb][neuron_nb]
        self.omegas = []    #  omega[layer_nb][neuron_nb]
        self.inputs.append( np.array(()) )
        self.weights.append( np.array(()) )    # input layer has no weights
        self.biases.append( np.array(()) )
        self.omegas.append( np.array(()) )
        for n in range(1, len(sizes)):
            self.inputs.append( np.zeros( sizes[n] ))
            self.biases.append( np.random.randn( sizes[n] ) )
        for n in range(1, len(sizes)):  # todo: fold back into one loop
            self.weights.append( np.random.randn( sizes[n], sizes[n-1]) )
            self.omegas.append( np.zeros( sizes[n] ) )
        
    def forward(self, data, label=None):
        assert(len(data) == len(self.outputs[0]))
        
        self.outputs[0] = np.array( data )
        
        for n in range(1, len(self.outputs)):
            # calculate layer inputs, all at once
            self.inputs[n] = np.dot( self.outputs[n-1], self.weights[n].T ) + self.biases[n]
            # calculate transfer function
            self.outputs[n] = self.funct(self.inputs[n])
            
        if label is not None:
            return np.sum(np.square(self.outputs[-1] - label))
            
    def SGD(self, data_vec, label_vec, learn_rate = 0.1, batch_size = 20):
        c = list(zip(data_vec, label_vec))
        np.random.shuffle(c)
        data_shuffled, label_shuffled = zip(*c)
        for k in range(0, len(data_shuffled), batch_size):
            data_batch = data_shuffled[k:k+batch_size]
            label_batch = label_shuffled[k:k+batch_size]
            self.train_batch(data_batch, label_batch, learn_rate)
        
           
    def train_batch(self, data_batch, label_batch, learn_rate = 0.1):
        assert(len(data_batch) == len(label_batch))
        
        delta_w, delta_b = self.backward(data_batch[0], label_batch[0], learn_rate)
        
        for i in range(1, len(data_batch)):
            temp_delta_w, temp_delta_b = self.backward(data_batch[i], label_batch[i], learn_rate)
             
            for j in range(len(delta_w)):
                delta_w[j] += temp_delta_w[j]
                delta_b[j] += temp_delta_b[j]
                
                
        for l in range(1, len(self.outputs)):
            self.weights[l] += -learn_rate/len(data_batch) * delta_w[l]
            self.biases[l] += -learn_rate/len(data_batch) * delta_b[l]
            
    def backward(self, data, label, learn_rate = 0.1):
        assert(len(data) == len(self.outputs[0]))
        assert(len(label) == len(self.outputs[-1]))
        
        self.forward(data)
        
        err_p = self.outputs[-1] - label
        sig_p = self.funct(self.inputs[-1], True)
        self.omegas[-1] = err_p * sig_p
        
        delta_w = []
        delta_b = []
                
        delta_w.append( np.outer( self.omegas[-1], self.outputs[-2] ) )
        delta_b.append( self.omegas[-1] )
                
        for l in range( len(self.outputs)-2, 0, -1 ):
            err_p = np.dot( self.omegas[l+1], self.weights[l+1] )   # do not transpose here
            sig_p = self.funct(self.inputs[l], True)
            self.omegas[l] = err_p * sig_p
            
            delta_w.insert(0, np.outer( self.omegas[l], self.outputs[l-1] ) )
            delta_b.insert(0, self.omegas[l] )
            
        delta_w.insert(0, np.array(()) )  # input layer doesn't have weiths, but we need to align arrays
        delta_b.insert(0, np.array(()) )
        
        return delta_w, delta_b
        
        #for l in range(1, len(self.outputs)):
        #    self.weights[l] += delta_w[l]
        #    self.biases[l] += delta_b[l]
        
        #self.weights[-1] += delta_w[-1]
        #self.biases[-1] += delta_b[-1]
        
    def eval_err(self, data_vec, label_vec):
        assert(len(data_vec) == len(label_vec))
        
        total_err = 0
        for i in range(len(data_vec)):
            total_err += self.forward( data_vec[i], label_vec[i] )
            
        return total_err
        
    def count_err(self, data_vec, label_vec):
        assert(len(data_vec) == len(label_vec))
        
        total_err = 0
        for i in range(len(data_vec)):
            self.forward( data_vec[i] )
            out = self.outputs[-1]
            #print('out=', out)
            label = np.argmax(out)
            #print('label=', label)
            if(label_vec[i][label] == 0):
                # should be one, wrong
                total_err += 1
                
            
        return total_err
            
    
    def vis_2D(self, layer_nb, neuron_nb, first = False):
        px = np.linspace(0, 1, 50)
        py = np.linspace(0, 1, 50)
        pz = np.zeros((len(px), len(px)))
        
        for i in range(len(px)):
            for j in range(len(py)):
                self.forward( (px[i], py[j]) )
                pz[i, j] = self.outputs[layer_nb][neuron_nb]
       
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
        
    def print_me(self, string=''):
        if len(string) != 0:
            print('  == ', string, ' ==')
            
        for n in range(len(self.outputs)):
            print(' == Layer ', n, ' == ')
            print('x:', self.inputs[n] )
            print('O:', self.outputs[n])
            print('W:', self.weights[n])
            print('B:', self.biases[n])
            print('o:', self.omegas[n])
