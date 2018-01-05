import numpy as np
import matplotlib.pyplot as plt
import pdb

from linear import LinearEnv
from agent import Agent


def test_run(nb_episodes, method,
            step_size, lmbda=None, e_rand=0.0):

    # States are encoded as:
    # STATE_ID       [0..1000]


    state_space = list(range(1001))
    
    action_space = [0, 1]  # move left, move right


    env = LinearEnv()
    agent = Agent(state_space=state_space,
                action_space=action_space,
                step_size=step_size,
                lmbda=lmbda,
                e_rand=e_rand)  

    RMSE = []    # root mean-squared error
    for e in range(nb_episodes):
        if e % 1000 == 0:
            print('episode:', e, '/', nb_episodes, '   ')


        obs = env.reset()
        agent.reset()

        agent.append_trajectory(t_step=0,
                                prev_action=None,
                                observation=obs,
                                reward=None,
                                done=None)


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
            elif method == 'td-lambda-online':
                agent.eval_td_lambda_online()
            if done:
                if method == 'mc-full':
                    agent.eval_mc_full()
                elif method == 'mc-offline':
                    agent.eval_mc_offline()
                elif method == 'td-offline':
                    agent.eval_td_offline()
                elif method == 'td-lambda-offline':
                    agent.eval_td_lambda_offline()


                #agent.print_trajectory()
                #pdb.set_trace()

                break

        arr_V = np.zeros(1001)
        arr_Q = np.zeros([1001, 2])
        for i in range(len(arr_V)):
            arr_V[i] = agent.V[i]
            arr_Q[i, 0] = agent.Q[i, 0]
            arr_Q[i, 1] = agent.Q[i, 1]

        rmse = np.sqrt(np.sum(np.power(
            LinearEnv.TRUE_VALUES - arr_V, 2)) / len(LinearEnv.TRUE_VALUES))
        RMSE.append(rmse)
    
    return RMSE, arr_V, arr_Q




def multi_run(nb_runs, nb_episodes, world_size, method,
            step_size, nb_steps=None, lmbda=None):
    multi_RMSE = []
    multi_final_V = []
    multi_final_Q = []

    for run in range(nb_runs):
        RMSE, final_V, final_Q = test_run(nb_episodes, world_size, method, 
                                 step_size, nb_steps, lmbda)

        multi_RMSE.append(RMSE)
        multi_final_V.append(final_V)
        multi_final_Q.append(final_Q)

    return multi_RMSE, multi_final_V, multi_final_Q



class Data:
    def __init__(self, label):
        self.label = label
        self.x = []
        self.y = []



def test_single():
    nb_runs = 5
    nb_episodes = 200

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
        'lmbda':     1.0,     # Monte-Carlo
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
    #tests = [td_offline]

    for test in tests:
        np.random.seed(0)
        print(test['method'])

        test['RMSE'], test['final_V'], test['final_Q'] = multi_run(
            nb_runs=nb_runs, nb_episodes=nb_episodes, world_size=world_size, 
            method=test['method'], step_size=test['stepsize'],
            nb_steps=test['nb_steps'], lmbda=test['lmbda'])


    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(LinearEnvVQ.GROUND_TRUTH[world_size][1:-1], label='True V', color='black')
    
    for test in tests:

        label = test['method']
        for i in range(nb_runs):
            ax1.plot(test['final_V'][i][1:-1], label=label, color=test['color'], alpha=0.3)

            # for LEFT action (0) here -> ----v
            #ax1.plot(test['final_Q'][i][1:-1, 0], label=label, color=test['color'], alpha=1.0)

            # for RIGHT action (1) here -> ---v
            #ax1.plot(test['final_Q'][i][1:-1, 1], label=label, color=test['color'], alpha=1.0)
    
            # average between two actions (should be the same as final_V above)
            final_V_from_Q = np.mean(test['final_Q'][i], axis=1)
            ax1.plot(final_V_from_Q[1:-1], label=label, color=test['color'], alpha=1.0)        

            ax2.plot(test['RMSE'][i], label=label, color=test['color'], alpha=0.3)
    
            label = None


    plt.legend()

    plt.grid()
    plt.show()




if __name__ == '__main__':
    # test_single()
    rmse, arr_V, arr_Q = \
        test_run(1000, 'td-lambda-offline', 0.1, lmbda=0.8, e_rand=1.0)

    # Load reference values if needed
    # Computed with 100,000 iterations of mc-full
    # ref_Q = np.load('ref_Q.npy')
    # ref_V = np.mean(ref_Q, axis=1)
    
    X = np.array(range(-500, 501))

    # arr_V = np.zeros(1001)
    # arr_Q = np.zeros([1001, 2])
    # for i in range(len(arr_V)):
    #     arr_V[i] = agent_V[i]
    #     arr_Q[i, 0] = agent_Q[i, 0]
    #     arr_Q[i, 1] = agent_Q[i, 1]
    arr_Vp = np.mean(arr_Q, axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(121)

    ax.plot(X, LinearEnv.TRUE_VALUES, color='gray', label='V-ref')
    ax.plot(X, arr_Vp, color='orange', label='V')
    # ax.plot(X, ref_Q[:,0], color='green', label='Q_left')
    # ax.plot(X, ref_Q[:,1], color='red', label='Q_right')
    ax.grid()

    
    ax = fig.add_subplot(122)
    ax.plot(rmse)

    plt.show()
