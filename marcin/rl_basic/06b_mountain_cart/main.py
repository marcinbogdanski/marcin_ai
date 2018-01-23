import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import time
import pdb

import gym
from mountain_car import MountainCarEnv

from agent import Agent

from logger import Logger, Log
from log_viewer import plot_mountain_car


def test_run(nb_episodes, nb_total_steps, expl_start,

    agent_discount,
    agent_nb_rand_steps,
    agent_e_rand_start,
    agent_e_rand_target,
    agent_e_rand_decay,

    mem_size_max,

    approximator, step_size, batch_size,
    ax_qmax_wf=None, ax_qmax_im=None, ax_policy=None, ax_trajectory=None,
    ax_stats=None, ax_q_series=None, logger=None):

    action_space = [0, 1, 2]  # move left, do nothing, move right

    # env = gym.make('MountainCar-v0')
    env = MountainCarEnv(log=logger.env)
    agent = Agent(action_space=action_space,
                discount=agent_discount,
                nb_rand_steps=agent_nb_rand_steps,
                e_rand_start=agent_e_rand_start,
                e_rand_target=agent_e_rand_target,
                e_rand_decay=agent_e_rand_decay,
                mem_size_max=mem_size_max,
                approximator=approximator,
                step_size=step_size,
                batch_size=batch_size,
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

            action = agent.pick_action(obs)

            agent.append_action(action=action)

            agent.log(episode, step, total_step)

            if total_step % 1000 == 0:
                print('e_rand', agent._epsilon_random, 
                    'step_size', agent._step_size)

                plot_mountain_car(logger, total_step, ax_qmax_wf, ax_qmax_im, 
                    ax_policy, ax_trajectory, ax_stats, ax_q_series)

                plt.pause(0.001)

            

            if total_step >= nb_total_steps:
                return

            agent.advance_one_step()

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
    
    ax_qmax_wf = fig.add_subplot(161, projection='3d')
    ax_qmax_im = fig.add_subplot(162)
    ax_policy = fig.add_subplot(163)
    ax_trajectory = fig.add_subplot(164)
    ax_stats = fig.add_subplot(165)
    ax_q_series = fig.add_subplot(166)

    test_run(
            nb_episodes=None,
            nb_total_steps=300000,
            expl_start=False,

            agent_discount=0.99,
            agent_nb_rand_steps=5000,
            agent_e_rand_start=1.0,
            agent_e_rand_target=0.1,
            agent_e_rand_decay=1.0/5000,

            mem_size_max=5000,

            approximator='neural',
            step_size=0.001,
            batch_size=128,
            ax_qmax_wf=ax_qmax_wf, 
            ax_qmax_im=ax_qmax_im,
            ax_policy=ax_policy,
            ax_trajectory=ax_trajectory, 
            ax_stats=ax_stats,
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


def main():
    logger = Logger()
    try:
        test_single(logger)
    finally:
        logger.save('data.log')
        print('log saved')


if __name__ == '__main__':
    main()
    
