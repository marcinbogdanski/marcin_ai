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
        plot_trajectory_2d(ax_traj, St_pos, St_vel, At)
        ax_traj.set_title('i=' + str(i))


        #
        #   Plot Q
        #
        if logger.q_val.data['q_val'][i] is not None:
            ax_q_max_wr.clear()
            plot_q_val_wireframe(ax_q_max_wr, logger.q_val.data['q_val'][i])
            ax_q_max_wr.set_title('i=' + str(i))

            ax_q_max_im.clear()
            plot_q_val_imshow(ax_q_max_im, logger.q_val.data['q_val'][i])
            ax_q_max_im.set_title('i=' + str(i))

            ax_pol.clear()
            plot_policy(ax_pol, logger.q_val.data['q_val'][i])
            ax_pol.set_title('i=' + str(i))
            
        plt.pause(0.1)

    plt.show()


def plot_trajectory_2d(ax, hpos, hvel, hact):
    hpos = np.array(hpos)
    hvel = np.array(hvel)
    hact = np.array(hact)

    data_back_p = []
    data_back_v = []
    data_stay_p = []
    data_stay_v = []
    data_fwd_p = []
    data_fwd_v = []

    for i in range(len(hpos)):
        if hact[i] == -1:
            data_back_p.append(hpos[i])
            data_back_v.append(hvel[i])
        elif hact[i] == 0:
            data_stay_p.append(hpos[i])
            data_stay_v.append(hvel[i])
        elif hact[i] == 1:
            data_fwd_p.append(hpos[i])
            data_fwd_v.append(hvel[i])
        elif hact[i] is None:
            # terminal state
            pass
        else:
            print('hact[i] = ', hact[i])
            raise ValueError('bad')


    ax.scatter(data_back_p, data_back_v, color='red', marker=',', lw=0, s=1)
    ax.scatter(data_stay_p, data_stay_v, color='blue', marker=',', lw=0, s=1)
    ax.scatter(data_fwd_p, data_fwd_v, color='green', marker=',', lw=0, s=1)

    ax.set_xlim([-1.2, 0.5])
    ax.set_ylim([-0.07, 0.07])


def plot_q_val_wireframe(ax, q_val):
    pos_size = q_val.shape[0]
    vel_size = q_val.shape[1]
    positions = np.linspace(-1.2, 0.49, pos_size)
    velocities = np.linspace(-0.07, 0.07, vel_size)
    actions = np.array([-1, 0, 1])

    Y, X = np.meshgrid(velocities, positions)
    Z = np.max(q_val, axis=2)

    ax.plot_wireframe(X, Y, Z)

    ax.set_xlabel('pos')
    ax.set_ylabel('vel')
    ax.set_zlabel('cost')

def plot_q_val_imshow(ax, q_val):

    Z = np.max(q_val, axis=2)
    z_min = np.min(q_val)
    z_max = np.max(q_val)

    extent=[-1.2, 0.5, -0.07, 0.07]
    ax.imshow(Z.T, extent=extent, 
        aspect='auto', origin='lower',
        interpolation='gaussian')

    ax.plot([-1.2, 0.5], [0, 0], color='black')
    ax.plot([-0.5, -0.5], [-0.07, 0.07], color='black')

def plot_policy(ax, q_val):
    pos_size = q_val.shape[0]
    vel_size = q_val.shape[1]
    positions = np.linspace(-1.2, 0.49, pos_size)
    velocities = np.linspace(-0.07, 0.07, vel_size)

    data_back_p = []
    data_back_v = []
    data_stay_p = []
    data_stay_v = []
    data_fwd_p = []
    data_fwd_v = []

    for pi in range(pos_size):
        for vi in range(vel_size):

            pos = positions[pi]
            vel = velocities[vi]

            q_back = q_val[pi, vi, 0]
            q_stay = q_val[pi, vi, 1]
            q_fwd = q_val[pi, vi, 2]

            max_act = np.argmax([q_back, q_stay, q_fwd]) - 1

            if max_act == -1:
                data_back_p.append(pos)
                data_back_v.append(vel)
            elif max_act == 0:
                data_stay_p.append(pos)
                data_stay_v.append(vel)
            elif max_act == 1:
                data_fwd_p.append(pos)
                data_fwd_v.append(vel)
            else:
                raise ValueError('bad')

    ax.scatter(data_back_p, data_back_v, color='red', marker='.')
    ax.scatter(data_stay_p, data_stay_v, color='blue', marker='.')
    ax.scatter(data_fwd_p, data_fwd_v, color='green', marker='.')

    ax.plot([-1.2, 0.5], [0, 0], color='black')
    ax.plot([-0.5, -0.5], [-0.07, 0.07], color='black')




if __name__ == '__main__':
    main()