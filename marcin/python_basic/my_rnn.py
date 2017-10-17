import numpy as np
import matplotlib.pyplot as plt
import pdb

# Hidden -> Output
Wy = np.array([[ 0.0],    # MEM neuron doesn't matter
               [ 1.0]])   # transfer ADD neuron 'as is'
By = np.array([[-0.5]])   # center step around 0.5

#             mem  add
S_ = np.array([[0, 0]])


Ww = np.array([[ 0.0, 1.0],
               [ 0.0, 0.0]])
Bw = np.array([[-0.5,-1.5]])

# Input -> Hidden
#                save to MEM neuron
#                     but do not affect ADD neuron
Wx = np.array([[ 1.0, 1.0]])

def sigmoid(x, deriv=False):
    if deriv:
        return np.multiply(sigmoid(x), (1 - sigmoid(x)))
    return 1 / (1 + np.exp(-x))

def step(x):
    return np.greater(x, 0).astype(np.float)


def run_nn(x):
    global S_
    S_ = step(np.dot(x, Wx) + np.dot(S_, Ww) + Bw)
    y_ = step(np.dot(S_, Wy) + By)
    return y_

def print_nn(strr, x, y):
    print('      IN   MEM  ADD  OUT')
    print('{0}: {1:.2f}  {2:.2f}  {3:.2f}  {4:.2f}'.format(strr, x, S_[0,0], S_[0,1], y))


xx = [0, 0, 1, 0, 1, 1, 1]
ss = []
tt = [0, 0, 0, 0, 0, 1, 1]
yy = []

#pdb.set_trace()

for x in xx:
    print('---')
    print_nn('bef', x, float('nan'))
    y = run_nn(x)
    print_nn('aft', float('nan'), y[0,0])
    yy.append(int(y[0,0]))


print('xx', xx)
# print('ss', ss)
print('tt', tt)
print('yy', yy)


def vis_2D(weights, biases):
        px = np.linspace(0, 1, 50)
        py = np.linspace(0, 1, 50)
        pz = np.zeros((len(px), len(px)))
        
        for i in range(len(px)):
            for j in range(len(py)):
                vec = np.array([[px[i], py[j]]])
                pz[i, j] = sigmoid(np.dot(vec, weights) + biases)
       
        
        extent = [px[0], px[-1], py[0], py[-1]]
        plt.imshow(pz.T, extent=extent, origin='lower')
        plt.colorbar()
        plt.show()
        


#vis_2D(Wy, By)