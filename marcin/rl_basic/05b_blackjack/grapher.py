import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import pdb

from logger import Logger

def plot_3d_wireframe(ax, Z, label, color):
    # ax.clear()

    dealer_card = list(range(1, 11))
    player_points = list(range(12, 22))
    X, Y = np.meshgrid(dealer_card, player_points)
    ax.plot_wireframe(X, Y, Z, label=label, color=color)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def plot_log_no_ace_3d(ax, log):
    maxx = np.greater(log.log_Q_no_ace_draw[-1], log.log_Q_no_ace_hold[-1])
    maxx = maxx.astype(float)

    plot_3d_wireframe(ax, log.log_Q_no_ace_hold[-1], 'hold', 'green')
    plot_3d_wireframe(ax, log.log_Q_no_ace_draw[-1], 'draw', 'red')
    #plot_3d_wireframe(ax, maxx, 'max', 'blue')

    ax.set_zlim(-1, 1)

def plot_ref_no_ace_3d(ax, log):
    plot_3d_wireframe(ax, log.ref_Q_no_ace_hold, 'hold', 'darkgreen')
    plot_3d_wireframe(ax, log.ref_Q_no_ace_draw, 'draw', 'darkred')
    ax.set_zlim(-1, 1)


def plot_log_ace_3d(ax, log):
    plot_3d_wireframe(ax, log.log_Q_ace_hold[-1], 'hold', 'green')
    plot_3d_wireframe(ax, log.log_Q_ace_draw[-1], 'draw', 'red')
    ax.set_zlim(-1, 1)

def plot_ref_ace_3d(ax, log):
    plot_3d_wireframe(ax, log.ref_Q_ace_hold, 'hold', 'darkgreen')
    plot_3d_wireframe(ax, log.ref_Q_ace_draw, 'draw', 'darkred')
    ax.set_zlim(-1, 1)


def main():
    log = Logger()
    log.load('td-lambda-offline.log')

    PLAYER_SUM = 12   # [12..21]
    DEALER_CARD = 10  # [1..10]

    x = log.log_t
    y = log.log_Q_no_ace_hold[:,PLAYER_SUM-12,DEALER_CARD-1]  # player_sum == 12, dealer_card == 10
    y2 = [log.ref_Q_no_ace_hold[PLAYER_SUM-12,DEALER_CARD-1] for _ in log.log_t]


    fig = plt.figure("No Ace")
    ax = fig.add_subplot(111, projection='3d')
    plot_ref_no_ace_3d(ax, log)
    plot_log_no_ace_3d(ax, log)

    fig = plt.figure("Ace")
    ax = fig.add_subplot(111, projection='3d')
    plot_ref_ace_3d(ax, log)
    plot_log_ace_3d(ax, log)

    fig = plt.figure("Num visits")
    ax = fig.add_subplot(111, projection='3d')
    plot_3d_wireframe(ax, log.log_Q_num_no_ace_hold, 'draw', 'green')
    plot_3d_wireframe(ax, log.log_Q_num_no_ace_draw, 'draw', 'red')
    
    print(log.log_Q_num_no_ace_draw.astype(int))
    print(log.log_Q_num_no_ace_hold.astype(int))

    plt.show()

    # pdb.set_trace()


if __name__ == '__main__':
    main()