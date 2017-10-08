import unittest
import neural
import neural_mini
import neural_tf
import random
import numpy as np

import backprop_nndl as nndl  # ground truth 


#IMPLEMENTATION = 'neural'
#IMPLEMENTATION = 'mini'
IMPLEMENTATION = 'tensor'
#IMPLEMENTATION = 'reference'

print('IMPLEMENTATION:', IMPLEMENTATION)

class NeuralTest(unittest.TestCase):

    def setUp(self):
       
        random.seed(0)
        np.random.seed(0)
        
        #
        #   Define weights for testing
        #
        weights_0 = np.array( [ [ 0.1, 0.4, 0.7 ],
                                [ 0.2, 0.5, 0.8 ] ] )
                                 
        biases_0 = np.array( [ [ 0.3, 0.6, 0.9 ] ] )
        
        weights_1 = np.array( [ [ 1.0 ], 
                                [ 1.1 ],
                                [ 1.2 ] ] )

        biases_1 = np.array( [ [ 1.3 ] ] )
        
        #
        #   Define test data
        #
        self.data_vec = [ (  np.array( [[0.1, 0.1]] ),  np.array( [[0]] )  ),
                          (  np.array( [[0.1, 0.9]] ),  np.array( [[1]] )  ),
                          (  np.array( [[0.9, 0.1]] ),  np.array( [[1]] )  ),
                          (  np.array( [[0.9, 0.9]] ),  np.array( [[0]] )  ) ]
                        
                        
        if IMPLEMENTATION == 'neural':
            self.nn = neural.NeuralNetwork( (2, 3, 1), init='norm' )
        elif IMPLEMENTATION == 'mini':
            self.nn = neural_mini.NeuralNetwork2( (2, 3, 1) )
        elif IMPLEMENTATION == 'tensor':
            neural_tf.reset_default_graph()
            self.nn = neural_tf.NeuralNetworkTF( (2, 3, 1) )
        elif IMPLEMENTATION == 'reference':
            self.nn = nndl.Network( (2, 3, 1) )  # reference neural network
            
            weights_0 = weights_0.T
            biases_0 = biases_0.T
            weights_1 = weights_1.T
            biases_1 = biases_1.T
            
            self.data_vec = [ (it[0].T, it[1]) for it in self.data_vec ]

        else:
            raise ValueError('Unknown implementation: ' + IMPLEMENTATION)
        
        # Make sure shapes match before asigning weights into neural network
        nn_weights_0 = self.nn.get_weights(0)
        nn_biases_0 = self.nn.get_biases(0)
        nn_weights_1 = self.nn.get_weights(1)
        nn_biases_1 = self.nn.get_biases(1)

        self.assertEqual( nn_weights_0.shape, weights_0.shape )
        self.assertEqual( nn_biases_0.shape, biases_0.shape )
        self.assertEqual( nn_weights_1.shape, weights_1.shape )
        self.assertEqual( nn_biases_1.shape, biases_1.shape )

        self.nn.set_weights(0, weights_0)
        self.nn.set_biases(0, biases_0)
        self.nn.set_weights(1, weights_1)
        self.nn.set_biases(1, biases_1)
    
    def tearDown(self):
        if IMPLEMENTATION == 'tensor':
            self.nn.close_tf_session()
        elif IMPLEMENTATION in ['neural', 'mini', 'reference']:
            pass
    
    def test_fun_sigmoid(self):

        if IMPLEMENTATION in ['neural', 'mini', 'tensor']:
            res1 = self.nn.fun_sigmoid( -1 )
            res2 = self.nn.fun_sigmoid( 0 )
            res3 = self.nn.fun_sigmoid( 1 )
            res4 = self.nn.fun_sigmoid( -1, deriv=True )
            res5 = self.nn.fun_sigmoid( 0, deriv=True )
            res6 = self.nn.fun_sigmoid( 1, deriv=True )
        elif IMPLEMENTATION == 'reference':
            res1 = nndl.sigmoid( -1 )
            res2 = nndl.sigmoid( 0 )
            res3 = nndl.sigmoid( 1 )
            res4 = nndl.sigmoid_prime( -1 )
            res5 = nndl.sigmoid_prime( 0 )
            res6 = nndl.sigmoid_prime( 1 )
        else:
            raise ValueError('Unknown implementation: ' + IMPLEMENTATION)
        
        self.assertAlmostEqual( res1, 0.2689414213699951, places=6 )
        self.assertAlmostEqual( res2, 0.5, places=6 )
        self.assertAlmostEqual( res3, 0.7310585786300049, places=6 )
        self.assertAlmostEqual( res4, 0.19661193324148185, places=6 )
        self.assertAlmostEqual( res5, 0.25, places=6 )
        self.assertAlmostEqual( res6, 0.19661193324148185, places=6 )


        
    def test_train_SGD(self):
    
        for i in range(1000):
            if IMPLEMENTATION in ['neural', 'mini', 'tensor']:
                self.nn.train_SGD(self.data_vec, 
                                  batch_size=2, eta=5.0)
            elif IMPLEMENTATION == 'reference':
                self.nn.SGD(self.data_vec, epochs=1, 
                            mini_batch_size=2, eta=5.0)
            else:
                raise ValueError('Unknown implementation: ' + IMPLEMENTATION)
                
            #if i % 10 == 0:
            #    if IMPLEMENTATION == 'neural':
            #        res = self.nn.evaluate(self.data_vec)
            #        #self.nn.vis_2D(1, 0)
            #    else:
            #        res = self.nn.eval_err(self.data_vec)
            #        #self.nn.vis_2D(2, 0)
            #    print(i, res)
                    
        
        if IMPLEMENTATION in ['neural', 'mini', 'tensor']:
            res, count = self.nn.evaluate(self.data_vec)
        elif IMPLEMENTATION == 'reference':
            res = self.nn.eval_err(self.data_vec)
        else:
            raise ValueError('Unknown implementation: ' + IMPLEMENTATION)
                
        # Due to undeterministic math using TF
        # we have to lower expected accuracy to 3 decimal places
        self.assertAlmostEqual( res, 0.0061221348573361678, places=3 )
        
    def test_train_batch(self):
                
        if IMPLEMENTATION in ['neural', 'mini', 'tensor']:
            self.nn.train_batch(self.data_vec, eta=0.3)
            weights_0 = self.nn.get_weights(0)
            biases_0 = self.nn.get_biases(0)
            weights_1 = self.nn.get_weights(1)
            biases_1 = self.nn.get_biases(1)
        elif IMPLEMENTATION == 'reference':
            self.nn.update_mini_batch(self.data_vec, eta=0.3)
            weights_0 = self.nn.get_weights(0).T
            biases_0 = self.nn.get_biases(0).T
            weights_1 = self.nn.get_weights(1).T
            biases_1 = self.nn.get_biases(1).T
        else:
            raise ValueError('Unknown implementation: ' + IMPLEMENTATION)
                
        self.assertAlmostEqual( weights_0[0][0], float('0.09966519') )
        self.assertAlmostEqual( weights_0[1][0], float('0.19966434') )
        self.assertAlmostEqual( weights_0[0][1], float('0.39973694') )
        self.assertAlmostEqual( weights_0[1][1], float('0.49973604') )
        self.assertAlmostEqual( weights_0[0][2], float('0.69982719') )
        self.assertAlmostEqual( weights_0[1][2], float('0.79982642') )
        self.assertAlmostEqual( weights_1[0][0], float('0.99794155') )
        self.assertAlmostEqual( weights_1[1][0], float('1.09754407') )
        self.assertAlmostEqual( weights_1[2][0], float('1.19725437') )
        
        self.assertAlmostEqual( biases_0[0][0], float('0.29918931'), places=6 )
        self.assertAlmostEqual( biases_0[0][1], float('0.59926552'), places=6 )
        self.assertAlmostEqual( biases_0[0][2], float('0.89939032'), places=6 )
        self.assertAlmostEqual( biases_1[0][0], float('1.29659672'), places=6 )
        
    def test_backward(self):
        
        data = self.data_vec[1][0]
        label = self.data_vec[1][1]
                
        if IMPLEMENTATION in ['neural', 'mini', 'tensor']:
            res_b, res_w = self.nn.backward( data, label )
        elif IMPLEMENTATION == 'reference':
            res_b, res_w = self.nn.backprop( data, label )  # reference impl.
            res_w = [ it.T for it in res_w ]  # transpose weight matrices
            res_b = [ it.T for it in res_b ]
        else:
            raise ValueError('Unknown implementation: ' + IMPLEMENTATION)
       
        
        self.assertEqual( len(res_w), 2 )
        self.assertIsInstance( res_w[0], np.ndarray )
        self.assertIsInstance( res_w[1], np.ndarray )
        self.assertEqual( res_w[0].shape, (2, 3) )
        self.assertEqual( res_w[1].shape, (3, 1) )
        
        self.assertEqual( len(res_b), 2 )
        self.assertIsInstance( res_b[0], np.ndarray )
        self.assertIsInstance( res_b[1], np.ndarray )
        self.assertEqual( res_b[0].shape, (1, 3) )
        self.assertEqual( res_b[1].shape, (1, 1) )
        
        self.assertAlmostEqual( res_w[0][0][0], float('-1.20024357e-05') )
        self.assertAlmostEqual( res_w[0][1][0], float('-1.08021921e-04') )
        self.assertAlmostEqual( res_w[0][0][1], float('-1.05535666e-05') )
        self.assertAlmostEqual( res_w[0][1][1], float('-9.49820993e-05') )
        self.assertAlmostEqual( res_w[0][0][2], float('-8.04044963e-06') )
        self.assertAlmostEqual( res_w[0][1][2], float('-7.23640467e-05') )
        self.assertAlmostEqual( res_w[1][0][0], float('-0.00031594') )
        self.assertAlmostEqual( res_w[1][1][0], float('-0.0003813') )
        self.assertAlmostEqual( res_w[1][2][0], float('-0.00043013') )
        
        self.assertAlmostEqual( res_b[0][0][0], float('-1.20024357e-04') )
        self.assertAlmostEqual( res_b[0][0][1], float('-1.05535666e-04') )
        self.assertAlmostEqual( res_b[0][0][2], float('-8.04044963e-05') )
        self.assertAlmostEqual( res_b[1][0][0], float('-0.0005095') )
        
    def test_forward(self):

        if IMPLEMENTATION in ['neural', 'mini', 'tensor']:
            data = np.array( [[0.1, 0.9]] )
            res = self.nn.forward(data)
        elif IMPLEMENTATION == 'reference':
            data = np.array( [[0.1],
                              [0.9]] )
            res = self.nn.feedforward(data)
        else:
            raise ValueError('Unknown implementation: ' + IMPLEMENTATION)
        
        self.assertIsInstance( res, np.ndarray )
        self.assertEqual( res.shape, (1, 1) )
        self.assertAlmostEqual( res[0][0], 0.977165764059, places=6 )
        
        
        if IMPLEMENTATION in ['neural', 'mini', 'tensor']:
            data = np.array( [[0.1, 0.9]] )
            res = self.nn.forward(data)
            self.assertIsInstance( res, np.ndarray )
            self.assertEqual( res.shape, (1, 1) )
            self.assertAlmostEqual( res[0][0], 0.977165764059, places=6 )
            
            data = np.array( [[0.1, 0.9], [0.9, 0.1]] )
            res = self.nn.forward(data)
            self.assertIsInstance( res, np.ndarray )
            self.assertEqual( res.shape, (2, 1) )
            self.assertAlmostEqual( res[0][0], 0.977165764059, places=6 )
            self.assertAlmostEqual( res[1][0], 0.976049023220, places=6 )
        elif IMPLEMENTATION == 'reference':
            pass # skip this test
        else:
            raise ValueError('Unknown implementation: ' + IMPLEMENTATION)

if __name__ == '__main__':
    unittest.main()
