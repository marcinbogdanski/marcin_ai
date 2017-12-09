import numpy as np
import matplotlib.pyplot as plt
import pdb

from linear import LinearEnv
from agent import Agent

def test_run(nb_episodes, world_size, method, step_size):
    env = LinearEnv(world_size)  # 2x terminal states will be added internally
    agent = Agent(world_size+2, step_size=step_size)  # add 2x terminal states

    RMS = []    # root mean-squared error
    for e in range(nb_episodes):

        # pdb.set_trace()

        obs = env.reset()
        agent.reset()
        agent.append_trajectory(t_step=0,
                                prev_action=None,
                                observation=obs,
                                reward=None,
                                done=None)

        # if e % 10 == 0:
        #     agent.step *= 0.95
        #     print(agent.step)

        while True:

            action = agent.pick_action(obs)

            #   ---   time step rolls here   ---

            obs, reward, done = env.step(action)

            agent.append_trajectory(t_step=env.t_step,
                        prev_action=action,
                        observation=obs,
                        reward=reward,
                        done=done)

            if method == 'TD':
                agent.eval_td_online()

            if done:
                if method == 'MC':
                    agent.eval_mc_offline()
                break

        rms = np.sqrt(np.sum(np.power(
            LinearEnv.GROUND_TRUTH[world_size] - agent.V, 2)) / 5)
        RMS.append(rms)

    return RMS, agent

def averaged_run(nb_runs, nb_episodes, world_size, method, step_size):
    outer_RMS = []

    for run in range(nb_runs):
        RMS, agent = test_run(nb_episodes, world_size, method, step_size)

        outer_RMS.append(RMS)
    
    outer_RMS = np.array(outer_RMS)
    average_RMS = np.sum(outer_RMS, axis=0) / len(outer_RMS)

    return average_RMS, agent

def test_MC_vs_TD():
    world_size = 5
    """This replicates Stutton and Barto (2017), Example 6.2 Random Walk"""
    avg_RMS_TD_15, agent = averaged_run(nb_runs=100, nb_episodes=100, world_size=world_size, 
                                        method='TD', step_size=0.15)
    avg_RMS_TD_10, agent = averaged_run(nb_runs=100, nb_episodes=100, world_size=world_size, 
                                        method='TD', step_size=0.10)
    avg_RMS_TD_05, agent = averaged_run(nb_runs=100, nb_episodes=100, world_size=world_size, 
                                        method='TD', step_size=0.05)

    avg_RMS_MC_01, agent = averaged_run(nb_runs=100, nb_episodes=100, world_size=world_size, 
                                        method='MC', step_size=0.01)
    avg_RMS_MC_02, agent = averaged_run(nb_runs=100, nb_episodes=100, world_size=world_size, 
                                        method='MC', step_size=0.02)
    avg_RMS_MC_03, agent = averaged_run(nb_runs=100, nb_episodes=100, world_size=world_size, 
                                        method='MC', step_size=0.03)


    print(np.round(LinearEnv.GROUND_TRUTH[world_size], 4))
    print(np.round(agent.V, 4))


    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(LinearEnv.GROUND_TRUTH[world_size][1:-1])
    ax.plot(agent.V[1:-1], label='TD')
    
    ax = fig.add_subplot(122)

    ax.plot(avg_RMS_TD_15, label='TD a=0.15')
    ax.plot(avg_RMS_TD_10, label='TD a=0.10')
    ax.plot(avg_RMS_TD_05, label='TD a=0.05')

    ax.plot(avg_RMS_MC_01, label='MC a=0.01')
    ax.plot(avg_RMS_MC_02, label='MC a=0.02')
    ax.plot(avg_RMS_MC_03, label='MC a=0.03')

    plt.legend()

    plt.grid()
    plt.show()


def test_n_step():
    world_size = 19
    """This replicates Stutton and Barto (2017), Example 6.2 Random Walk
    The only chang is that environment has reward -1 on far left side."""
    avg_RMS_n_step_15, agent = averaged_run(nb_runs=100, nb_episodes=10, world_size=world_size, 
                                        method='TD', step_size=0.15)



    print(np.round(LinearEnv.GROUND_TRUTH[world_size], 4))
    print(np.round(agent.V, 4))


    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(LinearEnv.GROUND_TRUTH[world_size][1:-1])
    ax.plot(agent.V[1:-1], label='TD')
    
    ax = fig.add_subplot(122)

    ax.plot(avg_RMS_TD_15, label='TD a=0.15')
    ax.plot(avg_RMS_TD_10, label='TD a=0.10')
    ax.plot(avg_RMS_TD_05, label='TD a=0.05')

    ax.plot(avg_RMS_MC_01, label='MC a=0.01')
    ax.plot(avg_RMS_MC_02, label='MC a=0.02')
    ax.plot(avg_RMS_MC_03, label='MC a=0.03')

    plt.legend()

    plt.grid()
    plt.show()



if __name__ == '__main__':
    test_MC_vs_TD()
    
