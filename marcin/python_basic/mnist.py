from __future__ import absolute_import, division, print_function, unicode_literals


import numpy as np
import matplotlib.pyplot as plt
import pickle


import sys, os, importlib
sys.path.append(os.getcwd())

import neural
import backprop_nndl as nndl

def load_data(filepath):
    import pickle, gzip
    
    with gzip.open(filepath, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        train_set, valid_set, test_set = u.load()
    
    return train_set, valid_set, test_set

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
    
def convert_data(data):
    training_inputs = [np.reshape(x, (784, 1)) for x in data[0]]
    training_results = [vectorized_result(y) for y in data[1]]
    return list(zip(training_inputs, training_results))
    
def convert_test_nndl(data):
    validation_inputs = [np.reshape(x, (784, 1)) for x in data[0]]
    return list(zip(validation_inputs, data[1]))
    

    
def test_mnist_nn():
    
    train_set, valid_set, test_set = load_data('..\mnist\mnist.pkl.gz')
    
    train_data = convert_data(train_set)
    valid_data = convert_data(valid_set)
    test_data = convert_data(test_set)
    
    #print('td', train_data[0:50] )
        
    nn = neural.NeuralNetwork( (784, 100, 10) )
    
    X = []
    E = []
    
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.plot(X, E)
    #plt.pause(0.01)
    
    
    
    for i in range(100):
        nn.train_SGD(train_data, batch_size=100, eta=6.0)
    
        train_err, train_count = nn.evaluate( train_data )
        valid_err, valid_count = nn.evaluate( valid_data )
        
        print('epoch:', i, 'trial_err:', train_err, 'train_count:', train_count, 'valid_err:', valid_err, 'valid_count:', valid_count, )
        
        #X.append(i)
        #E.append(valid_count)
        
        
        
        #ax.clear()
        #ax.plot(X, E)
        #plt.pause(0.01)
    
    #pickle.dump( nn, open('mnist.nn', 'wb') )
    
    #for i in range(len(train_labels)):
    #    val = train_labels[i]
    #    print('val', val)
    #    train_labels[i] = np.zeros( 10 )
    #    train_labels[i][val] = 1
    
    
def test_mnist_cust_nn():
    from scipy import misc

    num_img = misc.imread('number.png', flatten=True)
    num_img = 255 - num_img
    num_img /= 255
    
    nn = pickle.load( open('mnist.nn', 'rb') )
    nn.forward( num_img.flatten() )
    out = nn.outputs[-1]
    
    print('out:', out)
    print('answer:', np.argmax(out))
    
    train_set, valid_set, test_set = load_data('..\mnist\mnist.pkl.gz')
    
    train_data = train_set[0]
    train_labels = train_set[1]
    
    tt = np.reshape( train_set[0][150], (28, 28) )
    
    plt.figure()
    plt.imshow(num_img, cmap='gray_r')
    plt.show()
    
    return
    
def test_mnist_nndl():
    
    train_set, valid_set, test_set = load_data('..\mnist\mnist.pkl.gz')
    
    train_data = convert_data(train_set)
    validation_data = convert_test_nndl(valid_set)
    
    #train_data = train_data[:1000]
    #validation_data = validation_data[:100]
            
    print('td', validation_data[0][0].shape)
    print('tl', validation_data[0][1].shape)
    
    nn = nndl.Network( (784, 100, 10) )
    
    for i in range(10000):
        nn.SGD(train_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=validation_data )
    
        total_err = nn.eval_err( validation_data )
        print('total_err:', total_err)
    
    #for i in range(len(train_labels)):
    #    val = train_labels[i]
    #    print('val', val)
    #    train_labels[i] = np.zeros( 10 )
    #    train_labels[i][val] = 1
    
    
    tt = np.reshape( train_set[0][150], (28, 28) )
    print('tt', train_labels[150])
    
    plt.figure()
    plt.imshow(tt, cmap='gray_r')
    plt.show()
    
def main():
    #np.random.seed(40)
    test_mnist_nn()
    #test_mnist_nndl()

    #test_mnist_cust_nn()

if __name__ == '__main__':
    main()