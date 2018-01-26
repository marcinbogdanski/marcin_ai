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
    ax_q_max_wr = fig.add_subplot(151, projection='3d')
    ax_q_max_im = fig.add_subplot(152)
    ax_policy = fig.add_subplot(153)
    ax_trajectory = fig.add_subplot(154)
    ax_stats = None # fig.add_subplot(165)
    ax_q_series = fig.add_subplot(155)


    skip = 1000

    for total_step in range(0, len(logger.mem.total_steps), skip):
        print(total_step)

        plot_mountain_car(logger, total_step,
            ax_q_max_wr, ax_q_max_im, ax_policy, ax_trajectory,
            ax_stats, ax_q_series)

        plt.pause(0.1)

    plt.show()

def plot_mountain_car(logger, current_total_step, 
    ax_qmax_wf, ax_qmax_im, ax_policy, ax_trajectory, ax_stats, ax_q_series):
    extent = (-1, 0.5, -0.07, 0.07)

    if logger.q_val.data['q_val'][current_total_step] is not None:
        q_val = logger.q_val.data['q_val'][current_total_step]
        q_max = np.max(q_val, axis=2)

        if ax_qmax_wf is not None:
            ax_qmax_wf.clear()
            plot_q_val_wireframe(ax_qmax_wf, q_max,
                extent, ('pos', 'vel', 'q_max'))

        if ax_qmax_im is not None:
            ax_qmax_im.clear()
            plot_q_val_imshow(ax_qmax_im, q_max,
                extent, h_line=0.0, v_line=-0.5)
        
        if ax_policy is not None:
            ax_policy.clear()
            plot_policy(ax_policy, q_val,
                extent, h_line=0.0, v_line=-0.5)

    if ax_trajectory is not None:
        Rt_arr = logger.mem.data['Rt']
        St_pos_arr = logger.mem.data['St_pos']
        St_vel_arr = logger.mem.data['St_vel']
        At_arr = logger.mem.data['At']
        done = logger.mem.data['done']

        disp_len = 1000

        i = current_total_step
        Rt = Rt_arr[ max(0, i-disp_len) : i + 1 ]
        St_pos = St_pos_arr[ max(0, i-disp_len) : i + 1 ]
        St_vel = St_vel_arr[ max(0, i-disp_len) : i + 1 ]
        At = At_arr[ max(0, i-disp_len) : i + 1 ]

        ax_trajectory.clear()
        plot_trajectory_2d(ax_trajectory, 
            St_pos, St_vel, At, extent, h_line=0.0, v_line=-0.5)

    if ax_stats is not None:
        ax_stats.clear()
        i = current_total_step

        t_steps = logger.agent.total_steps[0:i:1]
        ser_e_rand = logger.agent.data['e_rand'][0:i:1]
        ser_rand_act = logger.agent.data['rand_act'][0:i:1]
        ser_mem_size = logger.agent.data['mem_size'][0:i:1]

        arr = logger.agent.data['rand_act'][max(0, i-1000):i]
        nz = np.count_nonzero(arr)
        print('RAND: ', nz, ' / ', len(arr))

        # ax_stats.plot(t_steps, ser_e_rand, label='e_rand', color='red')
        ax_stats.plot(t_steps, ser_rand_act, label='rand_act', color='blue')
        ax_stats.legend()

    if ax_q_series is not None:
        ax_q_series.clear()
        i = current_total_step
        t_steps = logger.q_val.total_steps[0:i:100]
        ser_E0 = logger.q_val.data['series_E0'][0:i:100]
        ser_E1 = logger.q_val.data['series_E1'][0:i:100]
        ser_E2 = logger.q_val.data['series_E2'][0:i:100]
        plot_q_series(ax_q_series, t_steps, ser_E0, ser_E1, ser_E2)


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

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])


def plot_trajectory_2d(ax, x_arr, y_arr, act_arr, extent, h_line, v_line):
    assert len(extent) == 4

    x_min, x_max, y_min, y_max = extent

    data_a0_x = []
    data_a0_y = []
    data_a1_x = []
    data_a1_y = []
    data_a2_x = []
    data_a2_y = []

    for i in range(len(x_arr)):
        if act_arr[i] == 0:
            data_a0_x.append(x_arr[i])
            data_a0_y.append(y_arr[i])
        elif act_arr[i] == 1:
            data_a1_x.append(x_arr[i])
            data_a1_y.append(y_arr[i])
        elif act_arr[i] == 2:
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

    ax.plot([x_min, x_max], [h_line, h_line], color='black')
    ax.plot([v_line, v_line], [y_min, y_max], color='black')

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

def plot_q_series(ax, t_steps, ser_0, ser_1, ser_2):

    # x = list(range(len(approx._q_back)))

    ax.plot(t_steps, ser_0, color='red')
    ax.plot(t_steps, ser_1, color='blue')
    ax.plot(t_steps, ser_2, color='green')


if __name__ == '__main__':
    main()