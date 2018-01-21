import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import collections
import argparse

from logger import Logger, Log

import pdb

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='Data log filename')
    args = parser.parse_args()

    logger = Logger()
    logger.load(args.filename)

    print(logger.env)
    print(logger.agent)
    print(logger.q_val)
    print(logger.mem)
    print(logger.approx)

    fig = plt.figure()
    ax_q_max_wr = fig.add_subplot(141, projection='3d')
    ax_q_max_im = fig.add_subplot(142)
    ax_pol = fig.add_subplot(143)
    ax_traj = fig.add_subplot(144)



    skip = 1000

    for i in range(0, len(logger.mem.total_steps), skip):
        print(i)

        extent = (-1, 0.5, -0.07, 0.07)

        #
        #   Plot trajectory
        #
        Rt_arr = logger.mem.data['Rt']
        St_pos_arr = logger.mem.data['St_pos']
        St_vel_arr = logger.mem.data['St_vel']
        At_arr = logger.mem.data['At']
        done = logger.mem.data['done']

        disp_len = 1000

        Rt = Rt_arr[ max(0, i-disp_len) : i + 1 ]
        St_pos = St_pos_arr[ max(0, i-disp_len) : i + 1 ]
        St_vel = St_vel_arr[ max(0, i-disp_len) : i + 1 ]
        At = At_arr[ max(0, i-disp_len) : i + 1 ]

        ax_traj.clear()
        plot_trajectory_2d(ax_traj, St_pos, St_vel, At, extent)
        ax_traj.set_title('i=' + str(i))


        #
        #   Plot Q
        #
        if logger.q_val.data['q_val'][i] is not None:

            q_val = logger.q_val.data['q_val'][i]
            q_max = np.max(q_val, axis=2)

            ax_q_max_wr.clear()
            plot_q_val_wireframe(ax_q_max_wr, q_max,
                extent, ('pos', 'vel', 'q_max'))
            ax_q_max_wr.set_title('i=' + str(i))

            ax_q_max_im.clear()
            plot_q_val_imshow(ax_q_max_im, q_max, 
                extent, h_line=0.0, v_line=-0.5)
            ax_q_max_im.set_title('i=' + str(i))

            ax_pol.clear()
            plot_policy(ax_pol, q_val,
                extent, h_line=0.0, v_line=-0.5)
            ax_pol.set_title('i=' + str(i))

        plt.pause(0.1)

    plt.show()


def plot_trajectory_2d(ax, x_arr, y_arr, act_arr, extent):
    assert len(extent) == 4

    x_min, x_max, y_min, y_max = extent

    data_a0_x = []
    data_a0_y = []
    data_a1_x = []
    data_a1_y = []
    data_a2_x = []
    data_a2_y = []

    for i in range(len(x_arr)):
        if act_arr[i] == -1:
            data_a0_x.append(x_arr[i])
            data_a0_y.append(y_arr[i])
        elif act_arr[i] == 0:
            data_a1_x.append(x_arr[i])
            data_a1_y.append(y_arr[i])
        elif act_arr[i] == 1:
            data_a2_x.append(x_arr[i])
            data_a2_y.append(y_arr[i])
        elif act_arr[i] is None:
            # terminal state
            pass
        else:
            print('act_arr[i] = ', act_arr[i])
            raise ValueError('bad')


    ax.scatter(data_a0_x, data_a0_y, color='red', marker=',', lw=0, s=1)
    ax.scatter(data_a1_x, data_a1_y, color='blue', marker=',', lw=0, s=1)
    ax.scatter(data_a2_x, data_a2_y, color='green', marker=',', lw=0, s=1)

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])


def plot_q_val_wireframe(ax, q_val, extent, labels):
    """Plot 2d q_val array on 3d wireframe plot.
    
    Params:
        ax - axis to plot on
        q_val - 2d numpy array as follows:
                1-st dim is X, increasing as indices grow
                2-nd dim is Y, increasing as indices grow
        extent - [x_min, x_max, y_min, y_max]
        labels - [x_label, y_label, z_label]
    """

    assert len(extent) == 4
    assert len(labels) == 3

    x_min, x_max, y_min, y_max = extent
    x_label, y_label, z_label = labels

    x_size = q_val.shape[0]
    y_size = q_val.shape[1]
    x_space = np.linspace(x_min, x_max, x_size)
    y_space = np.linspace(y_min, y_max, y_size)

    Y, X = np.meshgrid(y_space, x_space)
    
    ax.plot_wireframe(X, Y, q_val)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

def plot_q_val_imshow(ax, q_val, extent, h_line, v_line):
    assert len(extent) == 4

    x_min, x_max, y_min, y_max = extent

    ax.imshow(q_val.T, extent=extent, 
        aspect='auto', origin='lower',
        interpolation='gaussian')

    ax.plot([x_min, x_max], [h_line, h_line], color='black')
    ax.plot([v_line, v_line], [y_min, y_max], color='black')

def plot_policy(ax, q_val, extent, h_line, v_line):
    assert len(extent) == 4

    x_min, x_max, y_min, y_max = extent

    x_size = q_val.shape[0]
    y_size = q_val.shape[1]
    x_space = np.linspace(x_min, x_max, x_size)
    y_space = np.linspace(y_min, y_max, y_size)

    data_a0_x = []
    data_a0_y = []
    data_a1_x = []
    data_a1_y = []
    data_a2_x = []
    data_a2_y = []

    max_act = np.argmax(q_val, axis=2)

    for xi in range(x_size):
        for yi in range(y_size):

            x = x_space[xi]
            y = y_space[yi]

            if max_act[xi, yi] == 0:
                data_a0_x.append(x)
                data_a0_y.append(y)
            elif max_act[xi, yi] == 1:
                data_a1_x.append(x)
                data_a1_y.append(y)
            elif max_act[xi, yi] == 2:
                data_a2_x.append(x)
                data_a2_y.append(y)
            else:
                raise ValueError('bad')

    ax.scatter(data_a0_x, data_a0_y, color='red', marker='.')
    ax.scatter(data_a1_x, data_a1_y, color='blue', marker='.')
    ax.scatter(data_a2_x, data_a2_y, color='green', marker='.')

    ax.plot([x_min, x_max], [h_line, h_line], color='black')
    ax.plot([v_line, v_line], [y_min, y_max], color='black')




if __name__ == '__main__':
    main()