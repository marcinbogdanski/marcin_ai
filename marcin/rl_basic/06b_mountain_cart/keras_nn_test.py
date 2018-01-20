import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd



import neural_mini

import time
import pdb

hidden_size = 128
num_actions = 3

print('ver', keras.__version__)
print('backend', keras.backend.backend())
print('epsilon', keras.backend.epsilon())
print('floatx', keras.backend.floatx())
print('image data format', keras.backend.image_data_format())



# pdb.set_trace()

#################################

model = Sequential()
model.add(Dense(hidden_size, input_shape=(2, ), activation='sigmoid'))
model.add(Dense(hidden_size, activation='sigmoid'))
model.add(Dense(hidden_size, activation='sigmoid'))
model.add(Dense(hidden_size, activation='sigmoid'))
model.add(Dense(num_actions, activation='linear'))
model.compile(sgd(lr=.2), "mse")



data = np.array([[0.1, 0.2]])

model.predict(data)
model.predict(data)
model.predict(data)

start_time = time.time()
for i in range(10000):
    model.predict(data)
time_predict = time.time() - start_time

print('time_predict', time_predict)



##############################

nn = neural_mini.NeuralNetwork2([2, 128, 3])

start_time = time.time()
for i in range(10000):
    nn.forward(data)
time_predict = time.time() - start_time

print('time_predict', time_predict)