import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import time
import pdb

from mountain_car import MountainCarEnv
from agent import Agent

from logger import Logger, Log


def test_run(nb_episodes, nb_iterations, 
    approximator, step_size, e_rand, 
    axes=None, ax_pol=None, ax_hist=None, ax_w=None,
    ax_log=None, ax_q=None, logger=None):


    action_space = [-1, 0, 1]  # move left, do nothing, move right

    env = MountainCarEnv(log=logger.env)
    agent = Agent(action_space=action_space,
                approximator=approximator,
                step_size=step_size,
                e_rand=e_rand,
                log_agent=logger.agent,
                log_q_val=logger.q_val,
                log_mem=logger.mem,
                log_approx=logger.approx)

    #
    #   Initialise loggers
    #
    total_step = -1

    # RMSE = []    # root mean-squared error
    for episode in range(nb_episodes):
        
        step = 0
        total_step += 1

        print('episode:', episode, '/', nb_episodes, 'step', step, 'total_step', total_step)

        obs = env.reset()
        agent.reset()

        agent.append_trajectory(t_step=step,
                                observation=obs,
                                reward=None,
                                done=None)

        time_pick = 0

        while True:
            
            if agent._epsilon_random > 0.1:
                agent._epsilon_random -= 1.0 / 100000

            # print(i)

            time_start = time.time()
            action = agent.pick_action(obs)
            time_pick += time.time() - time_start

            agent.append_action(action=action)


            # print('AGENT logging', total_step, len(logger.q_val.data['q_val']))
            agent.log(episode, step, total_step)


            if step % 1000 == 0 and (axes is not None or ax_hist is not None):

                print('time_pick', time_pick)

                # agent._step_size *= 0.999

                print('e_rand', agent._epsilon_random, 'step_size', agent._step_size)

                if axes is not None:
                    for ax in axes:
                        if ax is not None:
                            ax.clear()
                    plot_approximator(
                        axes[0], axes[1], axes[2], axes[3], agent.Q)

                if ax_pol is not None:
                    ax_pol.clear()
                    plot_policy(ax_pol, agent.Q)

                if ax_hist is not None:
                    ax_hist.clear()
                    plot_history_2d(ax_hist, agent.Q._hist_pos, agent.Q._hist_vel,
                                          agent.Q._hist_act, agent.Q._hist_tar)



                # if ax_w is not None:
                #     if ax_w[0] is not None:
                #         ax_w[0].clear()
                #         plot_weights(ax_w[0], agent.Q._nn.weights_hidden)
                #     if ax_w[1] is not None:
                #         ax_w[1].clear()
                #         plot_weights(ax_w[1], agent.Q._nn.biases_hidden)
                #     if ax_w[2] is not None:
                #         ax_w[2].clear()
                #         plot_weights(ax_w[2], agent.Q._nn.weights_output)
                #     if ax_w[3] is not None:
                #         ax_w[3].clear()
                #         plot_weights(ax_w[3], agent.Q._nn.biases_output)

                # if ax_log is not None:
                #     ax_log.clear()
                #     #ax_log.plot(agent.Q._nn.w_abs_max, color='red')
                #     #ax_log.plot(agent.Q._nn.b_abs_max, color='green')
                #     #ax_log.plot(agent.Q._nn.w2_abs_max, color='orange')
                #     ax_log.plot(agent.Q._nn.b2_abs_max, color='blue')

                if ax_q is not None:
                    ax_q.clear()
                    plot_q_val(ax_q, agent.Q)

                plt.pause(0.001)

            
            # print('-------------')

            if total_step >= nb_iterations:
                return agent

            #   ---   time step rolls here   ---
            step += 1
            total_step += 1

            # print('episode:', episode, '/', nb_episodes, 'step', step, 'total_step', total_step)
            

            obs, reward, done = env.step(action)

            agent.append_trajectory(t_step=env.t_step,
                        observation=obs,
                        reward=reward,
                        done=done)

            agent.eval_td_online()
            
            if done or step >= 5000:
                if step >= 5000:
                    agent._epsilon_random = min(1.0, agent._epsilon_random + 1/10.0)
                print('espiode finished after iteration', step)
                agent.log(episode, step, total_step)
                break




    return agent






def test_single(logger):

    np.random.seed(0)

    nb_episodes = 1000
    nb_iterations = 50000

    
    logger.agent = Log('Agent')
    logger.q_val = Log('Q_Val')
    logger.env = Log('Environment', 'Mountain Car')
    logger.mem = Log('Memory', 'Replay buffer for DQN')
    logger.approx = Log('Approx', 'Approximator')




    fig = plt.figure()
    axb = None # fig.add_subplot(171, projection='3d')
    axs = None # fig.add_subplot(172, projection='3d')
    axf = None # fig.add_subplot(173, projection='3d')
    axm = fig.add_subplot(141, projection='3d')
    
    ax_pol = fig.add_subplot(142)
    
    ax_hist = fig.add_subplot(143)

    ax_w = None # fig.add_subplot(173)
    ax_w2 = None # fig.add_subplot(174)
    ax_b = None # fig.add_subplot(175)
    ax_b2 = None # fig.add_subplot(176)
    ax_log = None # fig.add_subplot(154)

    ax_q = fig.add_subplot(144)

    agent = test_run(nb_episodes=nb_episodes, nb_iterations=nb_iterations,
            approximator='neural', step_size=0.01, e_rand=0.1, 
            axes=[axb, axs, axf, axm], 
            ax_pol=ax_pol,
            ax_hist=ax_hist, 
            ax_w=[ax_w, ax_b, ax_w2, ax_b2], 
            ax_log=ax_log,
            ax_q=ax_q,
            logger=logger)

    plt.show()

def plot_weights(ax, weights):
    ax.hist(weights.flatten())

def plot_policy(ax, approx):
    positions = np.linspace(-1.2, 0.5, 16)
    velocities = np.linspace(-0.07, 0.07, 16)

    data_back_p = []
    data_back_v = []
    data_stay_p = []
    data_stay_v = []
    data_fwd_p = []
    data_fwd_v = []

    for pos in positions:
        for vel in velocities:
            q_back = approx.estimate((pos, vel), -1)
            q_stay = approx.estimate((pos, vel), 0)
            q_fwd = approx.estimate((pos, vel), 1)

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

def plot_history_2d(ax, hpos, hvel, hact, htar):
    hpos = np.array(hpos)
    hvel = np.array(hvel)
    hact = np.array(hact)
    htar = np.array(htar)

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
        else:
            raise ValueError('bad')


    ax.scatter(data_back_p, data_back_v, color='red', marker=',', lw=0, s=1)
    ax.scatter(data_stay_p, data_stay_v, color='blue', marker=',', lw=0, s=1)
    ax.scatter(data_fwd_p, data_fwd_v, color='green', marker=',', lw=0, s=1)

    ax.set_xlim([-1.2, 0.5])
    ax.set_ylim([-0.07, 0.07])


def plot_history_3d(ax, hpos, hvel, hact, htar):
    hpos = np.array(hpos)
    hvel = np.array(hvel)
    hact = np.array(hact)
    htar = np.array(htar)

    idx = hact==1

    r_pos = hpos[idx]
    r_vel = hvel[idx]
    r_tar = htar[idx]
    

    # ax.scatter(hpos[::10], hvel[::10], -htarget[::10])
    ax.scatter(r_pos, r_vel, r_tar)


def plot_q_val(ax, approx):

    est_q_back = approx.estimate((-0.5, 0.0), -1)
    est_q_stay = approx.estimate((-0.5, 0.0), 0)
    est_q_fwd = approx.estimate((-0.5, 0.0), 1) 

    print('estimates (0, 0):', est_q_back, est_q_stay, est_q_fwd)

    approx._q_back.append( est_q_back )
    approx._q_stay.append( est_q_stay )
    approx._q_fwd.append( est_q_fwd )

    # x = list(range(len(approx._q_back)))

    ax.plot(approx._q_back, color='red')
    ax.plot(approx._q_stay, color='blue')
    ax.plot(approx._q_fwd, color='green')


def plot_approximator(ax_back, ax_stay, ax_fwd, ax_max, approx):

    positions = np.linspace(-1.2, 0.5, 32)
    velocities = np.linspace(-0.07, 0.07, 32)
    actions = np.array([-1, 0, 1])

    X, Y = np.meshgrid(positions, velocities)
    Z_back = np.zeros_like(X)
    Z_stay = np.zeros_like(X)
    Z_fwd = np.zeros_like(X)
    Z_max = np.zeros_like(X)
    for x in range(32):
        for y in range(32):
            pos = X[x, y]
            vel = Y[x, y]
            
            q_back = approx.estimate((pos, vel), -1)
            q_stay = approx.estimate((pos, vel), 0)
            q_fwd = approx.estimate((pos, vel), 1)
            
            Z_back[x, y] = q_back
            Z_stay[x, y] = q_stay
            Z_fwd[x, y] = q_fwd
            arr = [q_back, q_stay, q_fwd]
            Z_max[x, y] = np.max(arr)
            
    axes = [ax_back, ax_stay, ax_fwd, ax_max]
    Z_vals = [Z_back,  Z_stay, Z_fwd, Z_max]
    colors = ['red', 'blue', 'green', 'black']

    for i in range(len(axes)):

        if axes[i] is not None:

            axes[i].plot_wireframe(X, Y, Z_vals[i], color=colors[i])

            axes[i].set_xlabel('pos')
            axes[i].set_ylabel('vel')
            axes[i].set_zlabel('cost')


def main():
    logger = Logger()
    try:
        test_single(logger)
    finally:
        logger.save('data.log')
        print('log saved')


if __name__ == '__main__':
    main()
    
