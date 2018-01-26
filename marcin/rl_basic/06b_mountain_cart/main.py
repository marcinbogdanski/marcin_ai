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
    ax_stats=None, ax_q_series=None, logger=None, 
    timing_arr=None, timing_dict=None):

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

    timing_arr.append('total')
    timing_arr.append('main_reset')
    timing_arr.append('main_agent_pick_action')
    timing_arr.append('main_agent_append_action')
    timing_arr.append('main_agent_log')
    timing_arr.append('main_plot')
    timing_arr.append('main_agent_advance_step')
    timing_arr.append('main_env_step')
    timing_arr.append('main_agent_append_trajectory')
    timing_arr.append('main_agent_td_online')
    timing_arr.append('  eval_td_start')
    timing_arr.append('  eval_td_get_batch')
    timing_arr.append('  eval_td_update')
    timing_arr.append('    update_loop')
    timing_arr.append('      update_loop_pred')
    timing_arr.append('    update_convert_numpy')
    timing_arr.append('    update_train_on_batch')
    timing_arr.append('    update2_create_arr')
    timing_arr.append('    update2_loop')
    timing_arr.append('    update2_scale')
    timing_arr.append('    update2_predict')
    timing_arr.append('    update2_post')
    timing_arr.append('    update2_train_on_batch')
    # timing_arr.append('hohohohoho')
    timing_dict.clear()
    for string in timing_arr:
        timing_dict[string] = 0

    #
    #   Initialise loggers
    #
    episode = -1
    total_step = -1
    while True:
        
        episode += 1
        if nb_episodes is not None and episode > nb_episodes:
            break

        if nb_total_steps is not None and total_step >= nb_total_steps:
            break

        time_total_start = time.time()
        
        step = 0
        total_step += 1

        print('episode:', episode, '/', nb_episodes,
            'step', step, 'total_step', total_step)

        time_start = time.time()

        # obs = env.reset()
        obs = env.reset(expl_start=expl_start)
        agent.reset(expl_start=expl_start)

        agent.append_trajectory(observation=obs,
                                reward=None,
                                done=None)

        timing_dict['main_reset'] += time.time() - time_start

        while True:

            time_start = time.time()
            action = agent.pick_action(obs)
            timing_dict['main_agent_pick_action'] += time.time() - time_start

            time_start = time.time()
            agent.append_action(action=action)
            timing_dict['main_agent_append_action'] += time.time() - time_start

            time_start = time.time()
            agent.log(episode, step, total_step)
            timing_dict['main_agent_log'] += time.time() - time_start


            time_start = time.time()
            if total_step % 1000 == 0:

                print()

                print('e_rand', agent._epsilon_random, 
                    'step_size', agent._step_size)

                print(str.upper(approximator))
                for key in timing_arr:
                    print(key, round(timing_dict[key], 3))

                if ax_qmax_wf is not None or ax_qmax_im is not None \
                    or ax_policy is not None or ax_trajectory is not None \
                    or ax_stats is not None or ax_q_series is not None:

                    plot_mountain_car(logger, total_step, ax_qmax_wf, ax_qmax_im, 
                        ax_policy, ax_trajectory, ax_stats, ax_q_series)
                    plt.pause(0.001)
            timing_dict['main_plot'] += time.time() - time_start
            

            if nb_total_steps is not None and total_step >= nb_total_steps:
                break

            time_start = time.time()
            agent.advance_one_step()
            timing_dict['main_agent_advance_step'] += time.time() - time_start

            #   ---   time step rolls here   ---
            step += 1
            total_step += 1

            time_start = time.time()
            obs, reward, done, _ = env.step(action)
            timing_dict['main_env_step'] += time.time() - time_start

            time_start = time.time()
            agent.append_trajectory(
                        observation=obs,
                        reward=reward,
                        done=done)
            timing_dict['main_agent_append_trajectory'] += time.time() - time_start

            time_start = time.time()
            agent.eval_td_online(timing_dict)
            timing_dict['main_agent_td_online'] += time.time() - time_start
            
            if done or step >= 200:
                print('espiode finished after iteration', step)
                time_start = time.time()
                agent.log(episode, step, total_step)
                timing_dict['main_agent_log'] += time.time() - time_start
                break

        timing_dict['total'] += time.time() - time_total_start

    return



def test_single(logger):

    # NOTE: there is another seed initialized to 0 on tensorflow import
    np.random.seed(0)

    
    logger.agent = Log('Agent')
    logger.q_val = Log('Q_Val')
    logger.env = Log('Environment', 'Mountain Car')
    logger.mem = Log('Memory', 'Replay buffer for DQN')
    logger.approx = Log('Approx', 'Approximator')

    timing_arr = []
    timing_dict = {}

    plotting_on = False

    if plotting_on:
        fig = plt.figure()
        axb = None # fig.add_subplot(171, projection='3d')
        axs = None # fig.add_subplot(172, projection='3d')
        axf = None # fig.add_subplot(173, projection='3d')
        ax_qmax_wf = fig.add_subplot(161, projection='3d')
        ax_qmax_im = fig.add_subplot(162)
        ax_policy = fig.add_subplot(163)
        ax_trajectory = fig.add_subplot(164)
        ax_stats = None # fig.add_subplot(165)
        ax_q_series = fig.add_subplot(166)
    else:
        axb = None
        axs = None
        axf = None
        ax_qmax_wf = None
        ax_qmax_im = None
        ax_policy = None
        ax_trajectory = None
        ax_stats = None
        ax_q_series = None

    approximator='keras'

    test_run(
            nb_episodes=None,
            nb_total_steps=2000,
            expl_start=True,

            agent_discount=0.99,
            agent_nb_rand_steps=64,
            agent_e_rand_start=1.0,
            agent_e_rand_target=0.1,
            agent_e_rand_decay=1.0/5000,

            mem_size_max=5000,

            approximator=approximator,
            step_size=0.001,
            batch_size=64,
            ax_qmax_wf=ax_qmax_wf, 
            ax_qmax_im=ax_qmax_im,
            ax_policy=ax_policy,
            ax_trajectory=ax_trajectory, 
            ax_stats=ax_stats,
            ax_q_series=ax_q_series,
            logger=logger,
            timing_arr=timing_arr,
            timing_dict=timing_dict)

    print()
    print(str.upper(approximator))
    for key in timing_arr:
        print(key, round(timing_dict[key], 3))

    if plotting_on:
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
    
