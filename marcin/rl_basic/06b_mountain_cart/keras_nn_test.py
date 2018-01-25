import numpy as np


import neural_mini

import time
import pdb



hidden_size = 256
num_actions = 3
learning_rate = 0.01



arr = []
arr2 = []
tar = []
tar2 = []
for i in range(500):
    inp = np.random.uniform(-1.0, 1.0, [128*5, 2]).astype(np.float32)
    arr.append(inp)
    inp2 = np.random.uniform(-1.0, 1.0, [128*5, 2]).astype(np.float32)
    arr2.append(inp2)
    tt = np.random.randint(0, 2, [128*5, 3]).astype(np.float32)
    tar.append(tt)
    tt2 = np.random.randint(0, 2, [128*5, 3]).astype(np.float32)
    tar2.append(tt2)

#################################
#       KERAS

# import keras
# from keras.models import Sequential
# from keras.layers.core import Dense
# from keras.optimizers import sgd
# print('ver', keras.__version__)
# print('backend', keras.backend.backend())
# print('epsilon', keras.backend.epsilon())
# print('floatx', keras.backend.floatx())
# print('image data format', keras.backend.image_data_format())

# model = Sequential()
# model.add(Dense(activation='sigmoid', input_dim=2, units=128))
# model.add(Dense(activation='linear', units=3))
# model.compile(loss='mse', optimizer=sgd(lr=learning_rate))


import tensorflow as tf

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(activation='sigmoid', input_dim=2, units=128))
model.add(tf.keras.layers.Dense(activation='linear', units=3))
model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=learning_rate))



print('KERAS')
start_time = time.time()
for i in range(500):
    model.predict(arr[i], batch_size=128*5)
    model.predict(arr2[i], batch_size=128*5)
time_predict = time.time() - start_time
print('  time_predict', time_predict)

start_time = time.time()
for i in range(500):
    model.train_on_batch(arr[i], tar[i])
    model.train_on_batch(arr2[i], tar2[i])
time_predict = time.time() - start_time
print('  time_train', time_predict)



#############################
#       TENSORFLOW

# import tensorflow as tf

# inputs = tf.placeholder(shape=[None, 2], dtype=tf.float32)
# targets = tf.placeholder(shape=[None, 3], dtype=tf.float32)
# nn = tf.layers.dense(inputs, 128, activation=tf.nn.sigmoid)
# nn = tf.layers.dense(nn, 3, activation=None)

# cost = tf.reduce_mean((nn-targets)**2)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# init = tf.global_variables_initializer()

# with tf.Session() as sess:
#     sess.run(init)
    
#     start_time = time.time()
#     for i in range(500):
#         val = sess.run([nn],
#             feed_dict={inputs:arr[i]})
#     time_predict = time.time() - start_time
#     print('  time_predict', time_predict)

#     start_time = time.time()
    
#     for i in range(500):
#         _, val = sess.run([optimizer, cost],
#             feed_dict={inputs:arr[i], targets:tar[i]})
    
#     time_train = time.time() - start_time
#     print('  time_train', time_train)

##############################

print('NUMPY')

nn = neural_mini.NeuralNetwork2([2, 128, 3])

start_time = time.time()
for i in range(500):
    nn.forward(arr[i])
    nn.forward(arr2[i])
time_predict = time.time() - start_time
print('  time_predict', time_predict)

start_time = time.time()
for i in range(500):
    nn.train_batch(arr[i], tar[i], eta=learning_rate)
    nn.train_batch(arr2[i], tar2[i], eta=learning_rate)
time_train = time.time() - start_time
print('  time_train', time_train)




