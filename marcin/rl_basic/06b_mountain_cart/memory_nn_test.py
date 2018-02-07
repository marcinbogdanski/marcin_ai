
import numpy as np
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import log_viewer
import agent

import tensorflow as tf

np.random.seed(0)
tf.set_random_seed(0)



def print_memory():

    mem = np.load('memory.npz')
    states_all = mem['states']
    actions_all = mem['actions']
    rewards_1_all = mem['rewards_1']
    states_1_all = mem['states_1']
    dones_all = mem['dones']


    fig = plt.figure()
    ax = fig.add_subplot(111)



    # plot_full_mem(ax, states_all, actions_all)
    # plot_tails(ax, states_all, actions_all)

    ka = agent.AggregateApproximator(0.3, [0, 1, 2], init_val=-100)
    plot_error(ax, ka, states_all, actions_all, rewards_1_all, states_1_all, dones_all)

    plt.show()


def plot_error(ax, ka, states, actions, rewards_1, states_1, dones):

    est_all = ka.estimate_all(states)

    est_all_1 = ka.estimate_all(states_1)

    errors_all = np.zeros([len(states)])

    for i in range(len(states)):
        pos = states[i, 0]
        vel = states[i, 1]
        act = actions[i, 0]
        rew = rewards_1[i, 0]
        est = est_all[i, act]
        done = dones[i, 0]

        if done:
            Tt = rew
        else:
            Tt = rew + 0.99 * max(est_all_1[i])

        error = Tt - est

        errors_all[i] = error

    ax.plot(errors_all)
        



def test_train():

    mem = np.load('memory.npz')
    states_all = mem['states'] #[0:2000]
    actions_all = mem['actions'] #[0:2000]
    rewards_1_all = mem['rewards_1'] #[0:2000]
    states_1_all = mem['states_1'] #[0:2000]
    dones_all = mem['dones'] #[0:2000]
    errors_all = np.zeros([len(states_all)]) + 1000 # arbitrary large error
    # indices = np.where(dones_all)[0]
    # errors_all[indices] = 99

    timing_dict = {}
    timing_dict['    update2_create_arr'] = 0
    timing_dict['    update2_scale'] = 0
    timing_dict['    update2_predict'] = 0
    timing_dict['    update2_post'] = 0
    timing_dict['    update2_train_on_batch'] = 0

    # ka = agent.AggregateApproximator(0.3, [0, 1, 2], init_val=-100)
    # ka = agent.TileApproximator(0.3, [0, 1, 2], init_val=-100)
    ka = agent.KerasApproximator(0.1, 0.99, 1024)

    fig = plt.figure()
    
    ax_wf = fig.add_subplot(141, projection='3d')
    ax_im = fig.add_subplot(142)
    ax_pl = fig.add_subplot(143)
    ax_er = fig.add_subplot(144)

    total_step = -1
    while True:
        total_step += 1

        batch_len = 1024
        #
        #   Get batch
        #
        # indices = np.random.randint(
        #     low=0, high=len(states_all), size=batch_len, dtype=int)

        #
        #   Get batch priority
        #
        cdf = np.cumsum(errors_all+0.001)
        cdf = cdf / cdf[-1]
        values = np.random.rand(batch_len)
        indices = np.searchsorted(cdf, values)

        states = np.take(states_all, indices, axis=0)
        actions = np.take(actions_all, indices, axis=0)
        rewards_1 = np.take(rewards_1_all, indices, axis=0)
        states_1 = np.take(states_1_all, indices, axis=0)
        dones = np.take(dones_all, indices, axis=0)

        # if 1102 in indices:
        #     pdb.set_trace()

        #
        #   Update
        #
        errors = ka.update2(states, actions, rewards_1, states_1, dones, timing_dict)

        errors_all[indices] = np.abs(errors)

        # plot_error(ax, ka, states_all, actions_all, rewards_1_all, states_1_all, dones_all)

        

        #
        #   Plot
        #
        if total_step % 10 == 0:
            print('total_step', total_step, indices)
            positions = np.linspace(-1.2, 0.5, 64)
            velocities = np.linspace(-0.07, 0.07, 64)
            actions = np.array([0, 1, 2])

            num_tests = len(positions) * len(velocities)
            pi_skip = len(velocities)
            states = np.zeros([num_tests, 2])
            for pi in range(len(positions)):
                for vi in range(len(velocities)):
                    states[pi*pi_skip + vi, 0] = positions[pi]
                    states[pi*pi_skip + vi, 1] = velocities[vi]


            q_list = ka.estimate_all(states)
            q_val = np.zeros([len(positions), len(velocities), len(actions)])
            
            for si in range(len(states)):    
                pi = si//pi_skip
                vi = si %pi_skip
                q_val[pi, vi] = q_list[si]

            q_max = np.max(q_val, axis=2)

            extent = (-1.2, 0.5, -0.07, 0.07)

            ax_wf.clear()
            log_viewer.plot_q_val_wireframe(ax_wf, q_max, extent, ('pos', 'vel', 'q_max'))

            ax_im.clear()
            log_viewer.plot_q_val_imshow(ax_im, q_max, extent, 0.0, -0.5)

            ax_pl.clear()
            log_viewer.plot_policy(ax_pl, q_val, extent, 0.0, -0.5)

            ax_er.clear()
            plt.plot(errors_all)

            plt.pause(0.001)



if __name__ == '__main__':
    # print_memory()
    test_train()













def plot_full_mem(ax, states, actions):
    """Plots full memory in topdown 2d plot

    Params:
        states - 2d numpy array
        actions - 2d numpy array
    """

    x_arr = states_all[:,0]
    y_arr = states_all[:,1]
    act_arr = actions_all[:,0]

    extent = (-1.2, 0.5, -0.07, 0.07)
    log_viewer.plot_trajectory_2d(
        ax, x_arr, y_arr, act_arr, extent, 0, -0.5)

def plot_tails(ax, states, actions):

    indices = np.where(dones_all[:,0])[0]
    x_arr = states_all[:,0]
    y_arr = states_all[:,1]
    act_arr = actions_all[:,0]

    extent = (-1.2, 0.5, -0.07, 0.07)

    for idx in indices:
        log_viewer.plot_trajectory_2d(
            ax,
            x_arr[idx-100:idx+1],
            y_arr[idx-100:idx+1],
            act_arr[idx-100:idx+1], extent, 0, -0.5)