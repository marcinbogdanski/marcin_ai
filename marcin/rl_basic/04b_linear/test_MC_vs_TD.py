import numpy as np
import matplotlib.pyplot as plt
import pdb

from linear import LinearEnv
from agent import Agent

def test_run(nb_episodes, world_size, method, step_size, n_steps=None):
    env = LinearEnv(world_size)  # 2x terminal states will be added internally
    agent = Agent(world_size+2, step_size=step_size)  # add 2x terminal states

    RMSE = []    # root mean-squared error
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

            if method == 'TD-online':
                agent.eval_td_online()

            if done:
                if method == 'MC-offline':
                    agent.eval_mc_offline()
                elif method == 'n-step-offline':
                    agent.eval_nstep_offline(n_steps)
                elif method == 'TD-offline':
                    agent.eval_td_offline()
                break

        rms = np.sqrt(np.sum(np.power(
            LinearEnv.GROUND_TRUTH[world_size] - agent.V, 2)) / world_size)
        RMSE.append(rms)

    return RMSE, agent.V

def multi_run(nb_runs, nb_episodes, world_size, method, step_size, n_steps=None):
    multi_RMSE = []
    multi_final_V = []

    for run in range(nb_runs):
        RMSE, final_V = test_run(nb_episodes, world_size, method, step_size, n_steps)

        multi_RMSE.append(RMSE)
        multi_final_V.append(final_V)

    return multi_RMSE, multi_final_V

def test_MC_vs_TD():
    """This replicates Stutton and Barto (2017), Example 6.2 Random Walk 5
    The only chang is that environment has reward -1 on far left side."""

    world_size = 5

    RSME_TD_15, final_V_TD_15 = multi_run(
        nb_runs=100, nb_episodes=100, world_size=world_size, 
        method='TD-online', step_size=0.15)
    RSME_TD_10, final_V_TD_10 = multi_run(
        nb_runs=100, nb_episodes=100, world_size=world_size, 
        method='TD-online', step_size=0.10)
    RSME_TD_05, final_V_TD_05 = multi_run(
        nb_runs=100, nb_episodes=100, world_size=world_size, 
        method='TD-online', step_size=0.05)

    RSME_MC_01, final_V_MC_01 = multi_run(
        nb_runs=100, nb_episodes=100, world_size=world_size, 
        method='MC-offline', step_size=0.01)
    RSME_MC_02, final_V_MC_02 = multi_run(
        nb_runs=100, nb_episodes=100, world_size=world_size, 
        method='MC-offline', step_size=0.02)
    RSME_MC_03, final_V_MC_03 = multi_run(
        nb_runs=100, nb_episodes=100, world_size=world_size, 
        method='MC-offline', step_size=0.03)

    RSME_TD_15_mean = np.mean(RSME_TD_15, 0)
    RSME_TD_10_mean = np.mean(RSME_TD_10, 0)
    RSME_TD_05_mean = np.mean(RSME_TD_05, 0)
    RSME_MC_01_mean = np.mean(RSME_MC_01, 0)
    RSME_MC_02_mean = np.mean(RSME_MC_02, 0)
    RSME_MC_03_mean = np.mean(RSME_MC_03, 0)

    #print(np.round(LinearEnv.GROUND_TRUTH[world_size], 4))
    #print(np.round(agent.V, 4))


    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(LinearEnv.GROUND_TRUTH[world_size][1:-1], color='black')
    ax.plot(final_V_TD_15[0][1:-1], label='TD a=0.15')
    ax.plot(final_V_TD_10[0][1:-1], label='TD a=0.10')
    ax.plot(final_V_TD_05[0][1:-1], label='TD a=0.05')
    ax.plot(final_V_MC_01[0][1:-1], label='MC a=0.01')
    ax.plot(final_V_MC_02[0][1:-1], label='MC a=0.02')
    ax.plot(final_V_MC_03[0][1:-1], label='MC a=0.03')
    
    ax = fig.add_subplot(122)
    ax.plot(RSME_TD_15_mean, label='TD a=0.15')
    ax.plot(RSME_TD_10_mean, label='TD a=0.10')
    ax.plot(RSME_TD_05_mean, label='TD a=0.05')
    ax.plot(RSME_MC_01_mean, label='MC a=0.01')
    ax.plot(RSME_MC_02_mean, label='MC a=0.02')
    ax.plot(RSME_MC_03_mean, label='MC a=0.03')

    plt.legend()

    plt.grid()
    plt.show()

class Data:
    def __init__(self, label):
        self.label = label
        self.x = []
        self.y = []

def test_n_step():
    """This replicates Stutton and Barto (2017), Example 7.2 Random Walk 19"""
    
    world_size = 19
    
    data = {}

    for n in [1, 2, 4, 8, 16]:
        print('n = ', n)
        temp = Data(str(n))

        for step_size in np.arange(0.1, 1.0, 0.1):
            RMSE, final_V = multi_run(
                nb_runs=100, nb_episodes=10, world_size=world_size, 
                method='n-step-offline', step_size=step_size, n_steps=n)

            RMSE_ep_mean = np.mean(RMSE, axis=1)
            RMSE_run_mean = np.mean(RMSE_ep_mean)

            temp.x.append(step_size)
            temp.y.append(RMSE_run_mean)

        data[n] = temp


    #print(np.round(LinearEnv.GROUND_TRUTH[world_size], 4))
    #print(np.round(agent.V, 4))


    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax.plot(LinearEnv.GROUND_TRUTH[world_size][1:-1], color='black')
    ax.plot(final_V[0][1:-1], label='TD')
    
    ax = fig.add_subplot(132)

    for key, value in data.items():
        ax.plot(value.x, value.y, label=value.label)

    plt.legend()

    plt.grid()
    plt.show()


def test_single():
    nb_runs = 3
    nb_episodes = 100
    world_size = 19


    test_A = {
        'method':    'TD-offline',
        'stepsize':  0.1,
        'n_steps':   None,
        'color':     'orange'
    }

    test_B = {
        'method':    'MC-offline',
        'stepsize':  0.01,
        'n_steps':   None,
        'color':     'red'
    }

    test_C = {
        'method':    'n-step-offline',
        'stepsize':  0.05,
        'n_steps':   8,
        'color':     'blue'
    }

    tests = [test_A, test_B, test_C]

    for test in tests:
        print(test['method'])
        # test_A_RMSE, test_A_final_V = multi_run(
        #     nb_runs=10, nb_episodes=100, world_size=world_size, 
        #     method='n-step-offline', step_size=0.2, n_steps=1)
        test['RMSE'], test['final_V'] = multi_run(
            nb_runs=nb_runs, nb_episodes=nb_episodes, world_size=world_size, 
            method=test['method'], step_size=test['stepsize'], n_steps=test['n_steps'])


    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(LinearEnv.GROUND_TRUTH[world_size][1:-1], label='True V')
    
    for test in tests:

        for i in range(nb_runs):
            ax1.plot(test['final_V'][i][1:-1], label='', color=test['color'], alpha=0.3)
    
            ax2.plot(test['RMSE'][i][1:-1], label='', color=test['color'], alpha=0.3)
    
    #ax.plot(avg_RMS_TD_10, label='TD')
    #ax.plot(avg_RMS_n_step_10, label='n-step')


    plt.legend()

    plt.grid()
    plt.show()



if __name__ == '__main__':
    #test_MC_vs_TD()
    test_n_step()

    #test_single()
    
