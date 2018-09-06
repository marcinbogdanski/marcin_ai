from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import neural

import pdb

# ----- begining of data preparation by deep learning course -----

data_path = '../bike_sharing_dataset/hour.csv'

rides = pd.read_csv(data_path)

dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
data.head()

quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std
    
# Save data for approximately the last 21 days 
test_data = data[-21*24:]

# Now remove the test data from the data set 
data = data[:-21*24]

# Separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

# Hold out the last 60 days or so of the remaining data as a validation set
train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]

    
# ----- end of data preparation by deep learning course -----

# ----- begining of reference neural network implementation -----

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))
        
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))          
        
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        # Replace 0 with your sigmoid calculation.
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        #self.activation_function = sigmoid
                    
    
    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            #### Implement the forward pass here ####
            ### Forward pass ###
            # TODO: Hidden layer - Replace these values with your calculations.
            hidden_inputs = np.dot(X, self.weights_input_to_hidden) # signals into hidden layer
            hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

            # TODO: Output layer - Replace these values with your calculations.
            final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
            final_outputs = final_inputs # no activation funtion on final layer
            
            #### Implement the backward pass here ####
            ### Backward pass ###

            # TODO: Output error - Replace this value with your calculations.
            error = y - final_outputs # Output layer error is the difference between desired target and actual output.
            output_error_term = error # no activation funtion on final layer, so no derivative either
                        
            # TODO: Calculate the hidden layer's contribution to the error
            hidden_error = np.dot(output_error_term, self.weights_hidden_to_output.T)  # need to transpose weights
            
            # TODO: Backpropagated error terms - Replace these values with your calculations.
            hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
            
            # Weight step (input to hidden)
            delta_weights_i_h += np.outer(X[:, None], hidden_error_term)     # do outer products here
            # Weight step (hidden to output)
            delta_weights_h_o += np.outer(hidden_outputs, output_error_term) # and here

        # TODO: Update the weights - Replace these values with your calculations.
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step
 
    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer 
                
        return final_outputs

def MSE(y, Y):
    return np.mean((y-Y)**2)

# ----- end of reference neural network implementation -----

def test_reference_nn():
    import sys
    
    #random.seed(0)
    np.random.seed(0)

    ### Set the hyperparameters here ###
    iterations = 100000
    learning_rate = 0.08    # good: 1.0   def: 0.1
    hidden_nodes = 30    # good: 10
    output_nodes = 1
    batch_size = 128     # good: 512     def: 128

    N_i = train_features.shape[1]

    network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

    print('Starting')
    
    losses = {'train':[], 'validation':[]}
    for ii in range(iterations):
        
        # Go through a random batch of 128 records from the training data set
        batch = np.random.choice(train_features.index, size=batch_size)
        X, y = train_features.ix[batch].values, train_targets.ix[batch]['cnt']
                                 
        network.train(X, y)
       
        # Printing out the training progress
        train_loss = MSE(network.run(train_features).T, train_targets['cnt'].values)
        val_loss = MSE(network.run(val_features).T, val_targets['cnt'].values)
        
        sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(iterations)) \
                         + "% ... Training loss: " + str(train_loss)[:5] \
                         + " ... Validation loss: " + str(val_loss)[:5] \
                         + " lr: " + str(network.lr) + " bs: " + str(batch_size) )
        sys.stdout.flush()
        
        losses['train'].append(train_loss)
        losses['validation'].append(val_loss)

   
    
def test_my_nn():
    import sys

    #random.seed(0)
    np.random.seed(0)
    
    my_train_data = [ ( np.array([x]), np.asarray([[y]])) for x, y in \
                  zip(train_features.values, train_targets['cnt'].values)]

    ### Set the hyperparameters here ###
    iterations = 100000
    learning_rate = 0.1    # good: 1.0   def: 0.1
    hidden_nodes = 10    # good: 10
    output_nodes = 1
    batch_size = 128     # good: 512     def: 128

    N_i = train_features.shape[1]
    
    # Initialise my own neural network
    nn = neural.NeuralNetwork( (N_i, hidden_nodes, 1) )
    nn.activations[-1] = nn.fun_linear
    #nn.bias_mult = 0  # disable biases
    for l in range(nn.num_layers):
        nn.biases[l].fill(0)

    print('Starting')
    
    fig = plt.figure()
    ax = fig.gca()
    
    plt.grid()
    
    losses = {'nn_train':[], 'nn_validation':[]}
    for ii in range(iterations):
        
        # Go through a random batch of 128 records from the training data set
        temp_shuffled = my_train_data[:]
        np.random.shuffle(temp_shuffled)
        data_batch = temp_shuffled[0:batch_size]
        
        # Train the network
        nn.train_batch( data_batch, learning_rate )
        #nn.train_batch( data_batch, learning_rate, lmbda = 50.0, n=len(my_train_data) )
       
        if ii % 100 == 0:
            # Printing out the training progress
            nn_train_loss = MSE(nn.forward(train_features).T, train_targets['cnt'].values)
            nn_val_loss = MSE(nn.forward(val_features).T, val_targets['cnt'].values)
        
        sum_W = 0
        for l in range(nn.num_layers):
            sum_W += np.sum(nn.weights[l])
            
        #if ii % 100 == 0 and learning_rate > 0.001:
        #    learning_rate *= 0.997
        
        sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(iterations)) \
                         + "% ... NN Training loss: " + str(nn_train_loss)[:5] \
                         + " ... NN Validation loss: " + str(nn_val_loss)[:5] \
                         + " ... NN LR: " + str(learning_rate)[:5] \
                         + " ... NN Sum W: " + str(sum_W) )
        sys.stdout.flush()
        
        losses['nn_train'].append(nn_train_loss)
        losses['nn_validation'].append(nn_val_loss)
        
        if True and ii % 1000 == 0 and ii > 10:
        
            #mean = sum(losses['nn_validation']) / len(losses['nn_validation'])
        
            plt.cla()
            ax.set_xticks(np.arange(0, 100000, 1000))
            ax.set_yticks(np.arange(0, 1., 0.1))
            ax.set_ylim( [0, 0.4] )
            plt.grid()
            plt.plot(losses['nn_train'][10:], color=(0.5, 0.5, 1.0), linewidth=1.0)
            plt.plot(losses['nn_validation'][10:], color=(1.0, 0.5, 0.0), linewidth=1.0)
            plt.pause(0.001)
            
    
    pdb.set_trace()
    
def main():
    #test_reference_nn()
    test_my_nn()
    
if __name__ == '__main__':
    main()
    

