import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from logger import Logger, Log

import pdb

def main():

    logger = Logger()
    logger.load('data.log')

    print(logger.env)
    print(logger.agent)
    print(logger.mem)
    print(logger.approx)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(logger.agent.total_steps)):
        ax.clear()
        plot_q_val(ax, logger.agent.data['q_val'][i])
        plt.pause(0.1)

    plt.show()



def plot_q_val(ax, q_val):
    positions = np.linspace(-1.2, 0.49, 8)
    velocities = np.linspace(-0.07, 0.07, 8)
    actions = np.array([-1, 0, 1])

    Y, X = np.meshgrid(positions, velocities)
    Z = np.max(q_val, axis=2)

    ax.plot_wireframe(X, Y, Z)

    ax.set_xlabel('pos')
    ax.set_ylabel('vel')
    ax.set_zlabel('cost')

if __name__ == '__main__':
    main()