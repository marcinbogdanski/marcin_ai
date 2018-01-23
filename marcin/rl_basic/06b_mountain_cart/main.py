import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import time
import pdb

import gym
from mountain_car import MountainCarEnv

from agent import Agent

from logger import Logger, Log
from log_viewer import plot_q_val_wireframe
from log_viewer import plot_q_val_imshow
from log_viewer import plot_policy
from log_viewer import plot_trajectory_2d


def test_run(nb_episodes, nb_total_steps, expl_start,
    approximator, step_size, e_rand, 
    ax_qmax_wf=None, ax_qmax_im=None, ax_policy=None, ax_trajectory=None,
    ax_q_series=None, logger=None):


    action_space = [0, 1, 2]  # move left, do nothing, move right

    # env = gym.make('MountainCar-v0')
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
    episode = -1
    total_step = -1
    while True:
        episode += 1
        if nb_episodes is not None and episode > nb_episodes:
            break
        
        step = 0
        total_step += 1

        print('episode:', episode, '/', nb_episodes,
            'step', step, 'total_step', total_step)

        # obs = env.reset()
        obs = env.reset(expl_start=expl_start)
        agent.reset(expl_start=expl_start)

        agent.append_trajectory(observation=obs,
                                reward=None,
                                done=None)

        while True:
            
            if agent._epsilon_random > 0.1:
                agent._epsilon_random -= 1.0 / 100000

            action = agent.pick_action(obs)

            agent.append_action(action=action)

            agent.log(episode, step, total_step)

            if total_step % 1000 == 0:
                print('e_rand', agent._epsilon_random, 'step_size', agent._step_size)

                extent = (-1, 0.5, -0.07, 0.07)

                if logger.q_val.data['q_val'][-1] is not None:
                    q_val = logger.q_val.data['q_val'][-1]
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

                    i = total_step
                    Rt = Rt_arr[ max(0, i-disp_len) : i + 1 ]
                    St_pos = St_pos_arr[ max(0, i-disp_len) : i + 1 ]
                    St_vel = St_vel_arr[ max(0, i-disp_len) : i + 1 ]
                    At = At_arr[ max(0, i-disp_len) : i + 1 ]

                    ax_trajectory.clear()
                    plot_trajectory_2d(ax_trajectory, 
                        St_pos, St_vel, At, extent, h_line=0.0, v_line=-0.5)


                if ax_q_series is not None:
                    ax_q_series.clear()
                    plot_q_series(ax_q_series, agent.Q)

                plt.pause(0.001)

            

            if total_step >= nb_total_steps:
                return

            #   ---   time step rolls here   ---
            step += 1
            total_step += 1

            obs, reward, done, _ = env.step(action)

            agent.append_trajectory(
                        observation=obs,
                        reward=reward,
                        done=done)

            agent.eval_td_online()
            
            if done or step >= 200:
                print('espiode finished after iteration', step)
                agent.log(episode, step, total_step)
                break




    return






def test_single(logger):

    np.random.seed(0)

    
    logger.agent = Log('Agent')
    logger.q_val = Log('Q_Val')
    logger.env = Log('Environment', 'Mountain Car')
    logger.mem = Log('Memory', 'Replay buffer for DQN')
    logger.approx = Log('Approx', 'Approximator')




    fig = plt.figure()
    axb = None # fig.add_subplot(171, projection='3d')
    axs = None # fig.add_subplot(172, projection='3d')
    axf = None # fig.add_subplot(173, projection='3d')
    
    ax_qmax_wf = fig.add_subplot(151, projection='3d')
    ax_qmax_im = fig.add_subplot(152)
    ax_policy = fig.add_subplot(153)
    ax_trajectory = fig.add_subplot(154)
    ax_q_series = fig.add_subplot(155)

    test_run(
            nb_episodes=1500000,
            nb_total_steps=1500000,
            expl_start=True,
            approximator='neural',
            step_size=0.01,
            e_rand=0.1,
            ax_qmax_wf=ax_qmax_wf, 
            ax_qmax_im=ax_qmax_im,
            ax_policy=ax_policy,
            ax_trajectory=ax_trajectory, 
            ax_q_series=ax_q_series,
            logger=logger)

    plt.show()





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


def plot_q_series(ax, approx):

    est_q_back = approx.estimate(np.array([0.4, 0.035]), 0)
    est_q_stay = approx.estimate(np.array([0.4, 0.035]), 1)
    est_q_fwd = approx.estimate(np.array([0.4, 0.035]), 2) 

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
    
