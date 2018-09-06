import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import pdb

from logger import DataLogger, DataReference

def plot_3d_wireframe(ax, Z, label, color):
    # ax.clear()

    dealer_card = list(range(1, 11))
    player_points = list(range(12, 22))
    X, Y = np.meshgrid(dealer_card, player_points)
    ax.plot_wireframe(X, Y, Z, label=label, color=color)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')



def main():
    log = DataLogger()
    log.load('td-lambda-offline.log')
    ref = DataReference('reference.npy')

    PLAYER_SUM = 12   # [12..21]
    DEALER_CARD = 10  # [1..10]

    x = log.t
    # player_sum == 12, dealer_card == 10
    y = log.Q_no_ace_hold[:,PLAYER_SUM-12,DEALER_CARD-1]
    y2 = [ref.Q_no_ace_hold[PLAYER_SUM-12,DEALER_CARD-1] for _ in log.t]


    fig = plt.figure("Experimetn 1")
    ax = fig.add_subplot(121, projection='3d', title='No Ace')
    plot_3d_wireframe(ax, ref.Q_no_ace_hold, 'hold', (0.5, 0.7, 0.5, 1.0))
    plot_3d_wireframe(ax, ref.Q_no_ace_draw, 'draw', (0.7, 0.5, 0.5, 1.0))
    plot_3d_wireframe(ax, log.Q_no_ace_hold[-1], 'hold', 'green')
    plot_3d_wireframe(ax, log.Q_no_ace_draw[-1], 'draw', 'red')
    ax.set_zlim(-1, 1)

    ax = fig.add_subplot(122, projection='3d', title='Ace')
    plot_3d_wireframe(ax, ref.Q_ace_hold, 'hold', (0.5, 0.7, 0.5, 1.0))
    plot_3d_wireframe(ax, ref.Q_ace_draw, 'draw', (0.7, 0.5, 0.5, 1.0))
    plot_3d_wireframe(ax, log.Q_ace_hold[-1], 'hold', 'green')
    plot_3d_wireframe(ax, log.Q_ace_draw[-1], 'draw', 'red')
    ax.set_zlim(-1, 1)

    # fig = plt.figure("ps 12 dc 10")
    # ax = fig.add_subplot(111)
    # ax.plot(x, y, label='us')
    # ax.plot(x, y2, label='ref')
    # ax.legend()

    
    
    print(log.Q_num_no_ace_draw.astype(int))
    print(log.Q_num_no_ace_hold.astype(int))

    plt.show()

    # pdb.set_trace()


if __name__ == '__main__':
    main()