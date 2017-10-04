from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import pickle

import sys, os, importlib
sys.path.append(os.getcwd())

import backprop_marcin
import neural

import backprop_nndl as nndl

        

 


def test_and():
    X = np.array( ((0, 0), (0, 1), (1, 0), (1, 1)) )
    labels = np.array( (0, 0, 0, 1) )
    
    weights = np.array( (0.5, 1.5) )
    biases = np.array( -0.75 )
    
    #weights = np.random.rand(2) * 2 - 1
    #biases = np.random.rand(1) * 2 - 1
                        
    print('X:', X)
    print('labels:', labels)
    print('weights:', weights)
    print('biases:', biases)
   
    vis2D(weights, biases, True)
    
    learn_rate = 0.1
    
    for i in range(1000):
        for i in range(len(X)):
            x = X[i]
            y = tanh(np.dot(x, weights) + biases)
            err_p = y - labels[i]
            sig_p = tanh(np.dot(x, weights) + biases, True)
            delta_w = -learn_rate * err_p * sig_p * x
            
            weights += delta_w
            biases += -learn_rate * err_p * sig_p
    
        vis2D(weights, biases)
        total_err = np.sum(np.square(tanh(np.dot(X, weights) + biases) - labels))
        print('total_err:', total_err, weights, biases)
    
def test_and_nn(seed=None):
    nn = Network( (2, 1) )
    
    print('---')
    print(nn.weights[-1], nn.biases[-1])
    
    
    data_vec = np.array( ((0, 0), (0, 1), (1, 0), (1, 1)) )
    label_vec = np.array( ( (0,), (0,), (0,), (1,)) )
    
    nn.vis_2D(-1, 0, True)
    
    for i in range(10000):
        #for j in range(len(data_vec)):
            #nn.backward( data_vec[j], label_vec[j] )
            
            #nn.train_batch( (data_vec[j],), (label_vec[j],), 0.1 )
        
        nn.train_batch( data_vec, label_vec, 0.1 )
        
        if i % 100 == 0:
            nn.vis_2D(-1, 0)
            total_err = nn.eval_err( data_vec, label_vec )
            print('total_err:', total_err, nn.weights[-1], nn.biases[-1])
            
    nn.print_me()
    nn.vis_2D(-1, 0)
    
def test_and_nndl():
    nn = nndl.Network( (2, 1) )
    
    data_vec = (  
                  (  np.array([(0,),(0,)]),  np.array([(0,),])  ),
                  (  np.array([(0,),(1,)]),  np.array([(0,),])  ),
                  (  np.array([(1,),(0,)]),  np.array([(0,),])  ),
                  (  np.array([(1,),(1,)]),  np.array([(1,),])  ),
                )
    
    
    
    nn.vis_2D(-1, 0, True)
    
    for i in range(10000):
        #for j in range(len(data_vec)):
        #    #nn.backward( data_vec[j], label_vec[j] )
                    
        print(nn.weights[-1], nn.biases[-1])
        return
            
        if i % 100 == 0:
            
            nn.vis_2D(-1, 0)
            total_err = nn.eval_err( data_vec )
            print('total_err:', total_err) #, nn.weights[-1], nn.biases[-1])
            
    nn.vis_2D(-1, 0)
    
def test_and_or_nn():
    nn = Network( (2, 2) )
    nn.weights[1][0][0] = 0.5
    nn.weights[1][0][1] = 1.5
    nn.biases[1][0] = -0.75
    
    nn.weights[1][1][0] = 0.5
    nn.weights[1][1][1] = 1.5
    nn.biases[1][1] = -0.75
    
    data_vec = np.array( ((0, 0), (0, 1), (1, 0), (1, 1)) )
    label_vec = np.array( ( (-1, -1), (-1, 1), (-1, 1), (1, 1)) )
    
    nn.vis_2D(-1, 1, True)
    
    for i in range(1000):
        for i in range(len(data_vec)):
            nn.backward( data_vec[i], label_vec[i] )
            
        nn.vis_2D(-1, 1)
        total_err = nn.eval_err( data_vec, label_vec )
        print('total_err:', total_err, nn.weights[-1], nn.biases[-1])
            
    nn.print_me()
    nn.vis_2D(True)



def test_circ_nn():
    nn = neural.NeuralNetwork( (2, 8, 1) )
    
    # sigmoid
    data_vec = [ (   np.array( ((0.0,), (0.0,)) ),   np.array(( 0,),)   ),
                 (   np.array( ((0.0,), (0.5,)) ),   np.array(( 0,),)   ),
                 (   np.array( ((0.0,), (1.0,)) ),   np.array(( 0,),)   ),
                 (   np.array( ((0.5,), (1.0,)) ),   np.array(( 0,),)   ),
                 (   np.array( ((1.0,), (1.0,)) ),   np.array(( 0,),)   ),
                 (   np.array( ((1.0,), (0.5,)) ),   np.array(( 0,),)   ),
                 (   np.array( ((1.0,), (0.0,)) ),   np.array(( 0,),)   ),
                 (   np.array( ((0.5,), (0.0,)) ),   np.array(( 0,),)   ),
                 (   np.array( ((0.4,), (0.4,)) ),   np.array(( 1,),)   ),
                 (   np.array( ((0.6,), (0.6,)) ),   np.array(( 1,),)   ),
                 (   np.array( ((0.4,), (0.6,)) ),   np.array(( 1,),)   ),
                 (   np.array( ((0.6,), (0.4,)) ),   np.array(( 1,),)   ),
                 (   np.array( ((0.4,), (0.4,)) ),   np.array(( 1,),)   ),
                 (   np.array( ((0.6,), (0.6,)) ),   np.array(( 1,),)   ),
                 (   np.array( ((0.4,), (0.6,)) ),   np.array(( 1,),)   ),
                 (   np.array( ((0.6,), (0.4,)) ),   np.array(( 1,),)   )
                ]
                
    #nn.vis_2D(1, 0, True)

    X = []
    E = []
    
    for i in range(1000):
         
        # tanh
        #nn.SGD(data_vec, label_vec, 0.01, 4)
         
        # sigmoid
        nn.train_SGD(data_vec, batch_size=8, eta=10.0 )


             
        if i % 10 == 0:
            #nn.vis_2D(1, 0)
            total_err = nn.evaluate( data_vec )
            X.append(i)
            E.append(total_err)
            print('total_err:', total_err)
            
    nn.print_debug()
    #nn.vis_2D(1, 0, True)
    
    plt.plot(X, E)
    plt.ylim((0, 5))
    plt.show()
    
    pickle.dump( nn, open('circle.nn', 'wb') )
    
def test_circ_load_nn():
    nn = pickle.load( open('circle.nn', 'rb') )
    nn.vis_2D(2, 0, True)
    input("press any key to exit")
    
def test_circ_nndl():
    nn = nndl.Network( (2, 8, 1) )
               
    # sigmoid
    data_vec = [ (   np.array( ((0.0,), (0.0,)) ),   np.array(( 0,),)   ),
                 (   np.array( ((0.0,), (0.5,)) ),   np.array(( 0,),)   ),
                 (   np.array( ((0.0,), (1.0,)) ),   np.array(( 0,),)   ),
                 (   np.array( ((0.5,), (1.0,)) ),   np.array(( 0,),)   ),
                 (   np.array( ((1.0,), (1.0,)) ),   np.array(( 0,),)   ),
                 (   np.array( ((1.0,), (0.5,)) ),   np.array(( 0,),)   ),
                 (   np.array( ((1.0,), (0.0,)) ),   np.array(( 0,),)   ),
                 (   np.array( ((0.5,), (0.0,)) ),   np.array(( 0,),)   ),
                 (   np.array( ((0.4,), (0.4,)) ),   np.array(( 1,),)   ),
                 (   np.array( ((0.6,), (0.6,)) ),   np.array(( 1,),)   ),
                 (   np.array( ((0.4,), (0.6,)) ),   np.array(( 1,),)   ),
                 (   np.array( ((0.6,), (0.4,)) ),   np.array(( 1,),)   ),
                 (   np.array( ((0.4,), (0.4,)) ),   np.array(( 1,),)   ),
                 (   np.array( ((0.6,), (0.6,)) ),   np.array(( 1,),)   ),
                 (   np.array( ((0.4,), (0.6,)) ),   np.array(( 1,),)   ),
                 (   np.array( ((0.6,), (0.4,)) ),   np.array(( 1,),)   )
                ]
           
    
    nn.vis_2D(2, 0, True)
    
    print('===')
    #print(nn.weights[-1])
    #print(nn.biases)
    
    for i in range(10000):
       
        # sigmoid
        nn.SGD(data_vec, epochs=1, mini_batch_size=8, eta=1.5 )
        #nn.SGD(data_vec, epochs=1, mini_batch_size=len(data_vec), eta=0.1*16 )
        #nn.update_mini_batch(data_vec, 0.1*16)
        
        #print('---')
        print(nn.weights[-1])
        
        if i > 0:
            return
            
        if True: # i % 100 == 0:
            nn.vis_2D(2, 0)
            total_err = nn.eval_err( data_vec )
            print('total_err:', total_err)
            if i > 2:
                return
            
    nn.vis_2D(2, 0, True)
    

    
def test_and_nn2():
    nn = NeuralNetwork( (2, 2, 1) )

    nn.weights[0][0][0] = 10
    nn.weights[0][0][1] = -10
    nn.biases[0][0][0] = 5
    
    nn.weights[0][1][0] = -10
    nn.weights[0][1][1] = 10
    nn.biases[0][1][0] = 5
    
    nn.weights[1][0][0] = -5
    nn.weights[1][0][1] = -5
    nn.biases[1][0][0] = 8
    
    nn.print_debug()
    
    nn.forward( np.matrix( [(0,), (0,)] ) )

    nn.print_debug()
    
    return
    
    nn.backward( np.matrix( [(0,), (0,)] ), ((0,),) )
    
    #nn.print_debug()
    
    nn.vis_2D(1, 0)
    
    X = np.array( ((0, 0), (0, 1), (1, 0), (1, 1)) )
    labels = np.array( (0, 0, 0, 1) )
    
    

    
def main():
    print('hoho')
    
    #X = np.arange(-3, 3, 0.1)
    #Y = sigmoid(X, True)
    
    #plt.figure()
    #plt.plot(X, Y)
    #plt.show()
    
    #test_and()
    
    #test_and_nn()
    #test_and_nndl()
    
    #test_and_or_nn()
    
    test_circ_nn()
    #test_circ_load_nn()
    #test_circ_nndl()
    
    #test_and_nn2()
    
    
    pass
    
    
if __name__ == '__main__':
    main()