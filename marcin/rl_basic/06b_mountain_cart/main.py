import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import time
import pdb

from mountain_car import MountainCarEnv
from agent import Agent

from logger import Logger, Log
from log_viewer import plot_q_val_wireframe
from log_viewer import plot_q_val_imshow
from log_viewer import plot_policy
from log_viewer import plot_trajectory_2d


def test_run(nb_episodes, nb_iterations, 
    approximator, step_size, e_rand, 
    axes=None, ax_pol=None, ax_traj=None, ax_w=None,
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


            if total_step % 1000 == 0 and (axes is not None or ax_traj is not None):

                print('time_pick', time_pick)

                print('e_rand', agent._epsilon_random, 'step_size', agent._step_size)

                extent = (-1, 0.5, -0.07, 0.07)

                if logger.q_val.data['q_val'][-1] is not None:
                    q_val = logger.q_val.data['q_val'][-1]
                    q_max = np.max(q_val, axis=2)

                    ax_q_max = axes[3]
                    if ax_q_max is not None:
                        ax_q_max.clear()
                        plot_q_val_wireframe(axes[3], q_max,
                            extent, ('pos', 'vel', 'q_max'))
                    
                    if ax_pol is not None:
                        ax_pol.clear()
                        plot_policy(ax_pol, q_val,
                            extent, h_line=0.0, v_line=-0.5)

                if ax_traj is not None:
                    Rt_arr = logger.mem.data['Rt']
                    St_pos_arr = logger.mem.data['St_pos']
                    St_vel_arr = logger.mem.data['St_vel']
                    At_arr = logger.mem.data['At']
                    done = logger.mem.data['done']

                    disp_len = 1000

                    i = total_step
                    Rt = Rt_arr[ max(0, i-disp_len) : i + 1 ]
                    St_pos = St_pos_arr[ max(0, i-disp_len) : i + 1 ]
                    St_vel = St_vel_arr[ max(0, i-disp_len) : i + 1 ]
                    At = At_arr[ max(0, i-disp_len) : i + 1 ]

                    ax_traj.clear()
                    plot_trajectory_2d(ax_traj, St_pos, St_vel, At, extent,
                        h_line=0.0, v_line=-0.5)


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
    nb_iterations = 500000

    
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
    
    ax_traj = fig.add_subplot(143)

    ax_w = None # fig.add_subplot(173)
    ax_w2 = None # fig.add_subplot(174)
    ax_b = None # fig.add_subplot(175)
    ax_b2 = None # fig.add_subplot(176)
    ax_log = None # fig.add_subplot(154)

    ax_q = fig.add_subplot(144)

    agent = test_run(nb_episodes=nb_episodes, nb_iterations=nb_iterations,
            approximator='aggregate', step_size=0.3, e_rand=0.0, 
            axes=[axb, axs, axf, axm], 
            ax_pol=ax_pol,
            ax_traj=ax_traj, 
            ax_w=[ax_w, ax_b, ax_w2, ax_b2], 
            ax_log=ax_log,
            ax_q=ax_q,
            logger=logger)

    plt.show()

def plot_weights(ax, weights):
    ax.hist(weights.flatten())





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

    est_q_back = approx.estimate((0.4, 0.035), -1)
    est_q_stay = approx.estimate((0.4, 0.035), 0)
    est_q_fwd = approx.estimate((0.4, 0.035), 1) 

    print('estimates (0, 0):', est_q_back, est_q_stay, est_q_fwd)

    approx._q_back.append( est_q_back )
    approx._q_stay.append( est_q_stay )
    approx._q_fwd.append( est_q_fwd )

    # x = list(range(len(approx._q_back)))

    ax.plot(approx._q_back, color='red')
    ax.plot(approx._q_stay, color='blue')
    ax.plot(approx._q_fwd, color='green')



def main():
    logger = Logger()
    try:
        test_single(logger)
    finally:
        logger.save('data.log')
        print('log saved')


if __name__ == '__main__':
    main()
    
