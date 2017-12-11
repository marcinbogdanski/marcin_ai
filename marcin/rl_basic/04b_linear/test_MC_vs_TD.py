import numpy as np
import matplotlib.pyplot as plt
import pdb

from linear import LinearEnv
from agent import Agent

def test_run(nb_episodes, world_size, method,
            step_size, nb_steps=None, lmbda=None):
    env = LinearEnv(world_size)  # 2x terminal states will be added internally
    agent = Agent(world_size+2,       # add 2x terminal states
                step_size=step_size,
                nb_steps=nb_steps,
                lmbda=lmbda)  

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

            if method == 'td-online':
                agent.eval_td_online()

            if done:
                if method == 'mc-offline':
                    agent.eval_mc_offline()
                elif method == 'n-step-offline':
                    agent.eval_nstep_offline()
                elif method == 'td-offline':
                    agent.eval_td_offline()
                elif method == 'lambda-return-offline':
                    agent.eval_lambda_return_offline()
                elif method == 'td-lambda-offline':
                    agent.eval_td_lambda_offline()
                break

        rms = np.sqrt(np.sum(np.power(
            LinearEnv.GROUND_TRUTH[world_size] - agent.V, 2)) / world_size)
        RMSE.append(rms)

    return RMSE, agent.V

def multi_run(nb_runs, nb_episodes, world_size, method,
            step_size, nb_steps=None, lmbda=None):
    multi_RMSE = []
    multi_final_V = []

    for run in range(nb_runs):
        RMSE, final_V = test_run(nb_episodes, world_size, method, 
                                 step_size, nb_steps, lmbda)

        multi_RMSE.append(RMSE)
        multi_final_V.append(final_V)

    return multi_RMSE, multi_final_V

def test_MC_vs_TD():
    """This replicates Stutton and Barto (2017), Example 6.2 Random Walk 5
    The only chang is that environment has reward -1 on far left side."""

    world_size = 5

    RSME_TD_15, final_V_TD_15 = multi_run(
        nb_runs=100, nb_episodes=100, world_size=world_size, 
        method='td-online', step_size=0.15)
    RSME_TD_10, final_V_TD_10 = multi_run(
        nb_runs=100, nb_episodes=100, world_size=world_size, 
        method='td-online', step_size=0.10)
    RSME_TD_05, final_V_TD_05 = multi_run(
        nb_runs=100, nb_episodes=100, world_size=world_size, 
        method='td-online', step_size=0.05)

    RSME_MC_01, final_V_MC_01 = multi_run(
        nb_runs=100, nb_episodes=100, world_size=world_size, 
        method='mc-offline', step_size=0.01)
    RSME_MC_02, final_V_MC_02 = multi_run(
        nb_runs=100, nb_episodes=100, world_size=world_size, 
        method='mc-offline', step_size=0.02)
    RSME_MC_03, final_V_MC_03 = multi_run(
        nb_runs=100, nb_episodes=100, world_size=world_size, 
        method='mc-offline', step_size=0.03)

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
                nb_runs=10, nb_episodes=10, world_size=world_size, 
                method='n-step-offline', step_size=step_size, nb_steps=n)

            RMSE_ep_mean = np.mean(RMSE, axis=1)
            RMSE_run_mean = np.mean(RMSE_ep_mean)

            temp.x.append(step_size)
            temp.y.append(RMSE_run_mean)

        data[n] = temp

    fig = plt.figure()
    #ax = fig.add_subplot(131)
    #ax.plot(LinearEnv.GROUND_TRUTH[world_size][1:-1], color='black')
    #ax.plot(final_V[0][1:-1], label='TD')
    
    ax = fig.add_subplot(111)

    for key, value in data.items():
        ax.plot(value.x, value.y, label=value.label)

    plt.legend()

    plt.grid()
    plt.show()

def test_lambda_return():
    """This replicates Stutton and Barto (2017), Example 12.3 Random Walk 19"""
    
    world_size = 19
    
    data = {}

    for lmbda in [0, 0.4, 0.8, 0.9, .95]:
        print('lmbda = ', lmbda)
        temp = Data(str(lmbda))

        for step_size in np.arange(0.1, 1.1, 0.1):
            print('  step = ', step_size)
            RMSE, final_V = multi_run(
                nb_runs=1, nb_episodes=10, world_size=world_size, 
                method='lambda-return-offline', step_size=step_size, lmbda=lmbda)

            RMSE_ep_mean = np.mean(RMSE, axis=1)
            RMSE_run_mean = np.mean(RMSE_ep_mean)

            temp.x.append(step_size)
            temp.y.append(RMSE_run_mean)

        data[lmbda] = temp

    fig = plt.figure()
    # ax = fig.add_subplot(131)
    # ax.plot(LinearEnv.GROUND_TRUTH[world_size][1:-1], color='black')
    # ax.plot(final_V[0][1:-1], label='TD')
    
    ax = fig.add_subplot(111)

    for key, value in data.items():
        ax.plot(value.x, value.y, label=value.label)

    plt.legend()

    plt.grid()
    plt.show()


def test_single():
    nb_runs = 5
    nb_episodes = 100
    world_size = 19

    # Experiments tuned for world size 19
    td_offline = {
        'method':    'td-offline',
        'stepsize':  0.15,
        'nb_steps':  None,
        'lmbda':     None,
        'color':     'blue'
    }
    mc_offline = {
        'method':    'mc-offline',
        'stepsize':  0.01,
        'nb_steps':  None,
        'lmbda':     None,
        'color':     'red'
    }
    td_lambda_offline = {
        'method':    'td-lambda-offline',
        'stepsize':  0.15,
        'nb_steps':  None,
        'lmbda':     0.3,
        'color':     'orange'
    }
    tests = [td_offline, mc_offline, td_lambda_offline]

    td_offline = {
        'method':    'td-offline',
        'stepsize':  0.15,
        'nb_steps':  None,
        'lmbda':     None,
        'color':     'blue'
    }
    td_online = {
        'method':    'td-online',
        'stepsize':  0.15,
        'nb_steps':  None,
        'lmbda':     None,
        'color':     'red'
    }
    #tests = [td_offline, td_online]

    for test in tests:
        np.random.seed(0)
        print(test['method'])
        # test_A_RMSE, test_A_final_V = multi_run(
        #     nb_runs=10, nb_episodes=100, world_size=world_size, 
        #     method='N-step-offline', step_size=0.2, n_steps=1)
        test['RMSE'], test['final_V'] = multi_run(
            nb_runs=nb_runs, nb_episodes=nb_episodes, world_size=world_size, 
            method=test['method'], step_size=test['stepsize'], nb_steps=test['nb_steps'], lmbda=test['lmbda'])


    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(LinearEnv.GROUND_TRUTH[world_size][1:-1], label='True V', color='black')
    
    for test in tests:

        label = test['method']
        for i in range(nb_runs):
            ax1.plot(test['final_V'][i][1:-1], label=label, color=test['color'], alpha=0.3)
    
            ax2.plot(test['RMSE'][i], label=label, color=test['color'], alpha=0.3)
    
            label = None

    #ax.plot(avg_RMS_TD_10, label='TD')
    #ax.plot(avg_RMS_n_step_10, label='n-step')


    plt.legend()

    plt.grid()
    plt.show()


def test_test():

    RMSE, final_V = multi_run(
            nb_runs=1, nb_episodes=1, world_size=19, 
            method='lambda-return-offline', step_size=0.1, 
            nb_steps=None, lmbda=0.9)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(LinearEnv.GROUND_TRUTH[19][1:-1], label='True V')
    ax1.plot(final_V[0][1:-1])
    ax2 = fig.add_subplot(122)
    ax2.plot(RMSE[0])

    plt.show()


if __name__ == '__main__':
    #test_MC_vs_TD()
    #test_n_step()
    #test_lambda_return()

    test_single()
    
