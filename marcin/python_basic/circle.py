import numpy as np
import matplotlib.pyplot as plt
import pickle
import pdb

inputs = np.array([[0.0, 0.0],  # 1st training example,
                   [0.0, 0.5],  # 2nd training example,
                   [0.0, 1.0],  # and so on
                   [0.5, 1.0],
                   [1.0, 1.0],
                   [1.0, 0.5],
                   [1.0, 0.0],
                   [0.5, 0.0],
                   [0.4, 0.4],
                   [0.4, 0.6],
                   [0.6, 0.4],
                   [0.6, 0.6]], dtype=np.float32)

targets = np.array([[0],    # Target for 1st example,
                    [0],    # ...
                    [0], 
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [1],
                    [1],
                    [1],
                    [1]], dtype=np.float32)


learning_rate = 0.5



# Initial weights and biases
weights_hidden = np.random.randn(2, 2) * 0.1
biases_hidden = np.array([[ 0.1, 0.1 ]], dtype=np.float32)

weights_output = np.random.randn(2, 1) * 0.1
biases_output = np.array([[ 0.1 ]], dtype=np.float32)

def sigmoid(x, deriv=False):
    if deriv:
        return np.multiply(sigmoid(x), (1 - sigmoid(x)))
    return 1 / (1 + np.exp(-x))

def forward(inputs):
    global weights_hidden, biases_hidden, weights_output, biases_output

    layer_1 = sigmoid(np.dot(inputs, weights_hidden) + biases_hidden)

    output_layer = sigmoid(np.dot(layer_1, weights_output) + biases_output)

    return output_layer

def backward(inputs, targets, learning_rate):
    global weights, biases
    
    y = forward(inputs)

    err_p = y - targets

    d_weights = np.dot(inputs.T, err_p) 
    d_biasess = np.sum(err_p, keepdims=True)

    weights += -learning_rate * d_weights / len(inputs)
    biases += -learning_rate * d_biasess / len(inputs)

def map_input():
    """Runs NN on 2D input space (for plotting)"""
    px = np.linspace(0, 1, 50)
    py = np.linspace(0, 1, 50)
    pz = np.zeros((len(px), len(px)))
    
    for i in range(len(px)):
        for j in range(len(py)):
            x = [[ px[i], px[j] ]]
            result = forward(x)
            pz[i, j] = result[0, 0]

    return pz



error_over_time = []
value_map_over_time = {}

for i in range(1001):
    
    for j in range(len(inputs)):
        x = inputs[j:j+1]
        t = targets[j:j+1]
        y = backward(x, t, learning_rate)

    # calc error and save
    err = np.sum(np.square(forward(inputs) - targets))
    error_over_time.append(err)
    if i % 50 == 0 or i == 0:
        print('i: ', i, 'err:', err)   
        value_map_over_time[i] = map_input()


# Plotting only below

fig = plt.figure()
ax = fig.add_subplot(111)
ax.title.set_text('Error over time')
ax.set_xlabel('Epoch')
ax.set_ylabel('Error')
ax.set_ylim([0,2])
ax.plot(error_over_time)

extent = [0, 1, 0, 1]

fig = plt.figure()
ax = fig.add_subplot(221)
ax.title.set_text('Results at epoch 0')
ax.set_xlabel('input[0]')
ax.set_ylabel('input[1]')
value_map = value_map_over_time[0]
im = ax.imshow(value_map.T, vmin=0, vmax=1, extent=extent, origin='lower')
fig.colorbar(im)

ax = fig.add_subplot(222)
ax.title.set_text('Results at epoch 50')
ax.set_xlabel('input[0]')
ax.set_ylabel('input[1]')
value_map = value_map_over_time[50]
im = ax.imshow(value_map.T, vmin=0, vmax=1, extent=extent, origin='lower')
fig.colorbar(im)

ax = fig.add_subplot(223)
ax.title.set_text('Results at epoch 100')
ax.set_xlabel('input[0]')
ax.set_ylabel('input[1]')
value_map = value_map_over_time[100]
im = ax.imshow(value_map.T, vmin=0, vmax=1, extent=extent, origin='lower')
fig.colorbar(im)

ax = fig.add_subplot(224)
ax.title.set_text('Results at epoch 1000')
ax.set_xlabel('input[0]')
ax.set_ylabel('input[1]')
value_map = value_map_over_time[1000]
im = ax.imshow(value_map.T, vmin=0, vmax=1, extent=extent, origin='lower')
fig.colorbar(im)

plt.tight_layout()
plt.show()


