import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import neural_mini

import pdb



data_vec = [ (np.array([[0.0, 0.0]]), np.array([[0.0]])),
             (np.array([[0.0, 0.5]]), np.array([[0.0]])), 
             (np.array([[0.0, 1.0]]), np.array([[0.0]])), 
             (np.array([[0.5, 1.0]]), np.array([[0.0]])), 
             (np.array([[1.0, 1.0]]), np.array([[0.0]])), 
             (np.array([[1.0, 0.5]]), np.array([[0.0]])), 
             (np.array([[1.0, 0.0]]), np.array([[0.0]])), 
             (np.array([[0.5, 0.0]]), np.array([[0.0]])), 
             (np.array([[0.4, 0.4]]), np.array([[10.0]])), 
             (np.array([[0.4, 0.6]]), np.array([[10.0]])), 
             (np.array([[0.6, 0.4]]), np.array([[10.0]])), 
             (np.array([[0.6, 0.6]]), np.array([[10.0]])) ]




learning_rate = 0.01


def main():

    np.random.seed(0)

    nn = neural_mini.NeuralNetwork2([2, 128, 1])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    for i in range(10001):
        nn.train_SGD(data_vec, 12, eta=learning_rate)

        if i % 100 == 0:
            print(i)
            ax.clear()
            plot_nn(ax, nn)
            plt.pause(0.001)

    ax.clear()
    plot_nn(ax, nn)
    plt.show()


def plot_nn(ax, nn):

    x_vec = np.linspace(0.0, 1.0, 64)
    y_vec = np.linspace(0.0, 1.0, 64)
    
    X, Y = np.meshgrid(x_vec, y_vec)
    Z = np.zeros_like(X)

    for x in range(64):
        for y in range(64):
            xx = X[x, y]
            yy = Y[x, y]
                        
            val = nn.forward(np.array([[xx, yy]]))
                
            Z[x, y] = val
            
    ax.plot_wireframe(X, Y, Z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Y')




if __name__ == '__main__':
    main()