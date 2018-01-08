import numpy as np
import matplotlib.pyplot as plt
import pdb

from linear import LinearEnv
from agent import Agent


def test_run(nb_episodes, method, nb_states,
            step_size, lmbda=None, e_rand=0.0):

    # States are encoded as:
    # STATE_ID       [0..1000]


    state_space = list(range(1001))
    state_space.append('TERMINAL')
    
    action_space = [0, 1]  # move left, move right


    env = LinearEnv()
    agent = Agent(state_space=state_space,
                action_space=action_space,
                nb_states=nb_states,
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

            if done:
                obs = 'TERMINAL'  # Force unique terminal state

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
                break

        arr_V = np.zeros(1001)
        arr_Q = np.zeros([1001, 2])
        for i in range(len(arr_V)):
            arr_V[i] = agent.V[i]
            arr_Q[i, 0] = agent.Q[i, 0]
            arr_Q[i, 1] = agent.Q[i, 1]
        arr_Vp = np.mean(arr_Q, axis=1)

        rmse = np.sqrt(np.sum(np.power(
            LinearEnv.TRUE_VALUES - arr_Vp, 2)) / len(LinearEnv.TRUE_VALUES))
        RMSE.append(rmse)
    
    return RMSE, arr_V, arr_Q




def test_multi():
    nb_episodes = 100

    # Experiments tuned for world size 19
    mc_full = {
        'method':    'mc-full',
        'nb_states': 1001,
        'stepsize':  None,
        'lmbda':     None,
        'e_rand':    1.0,
        'color':     'gray'
    }
    td_lambda_offline_1000 = {
        'method':    'td-lambda-offline',
        'nb_states': 1001,
        'stepsize':  0.1,
        'lmbda':     0.8,
        'e_rand':    1.0,
        'color':     'orange'
    }
    td_lambda_offline_100 = {
        'method':    'td-lambda-offline',
        'nb_states': 101,
        'stepsize':  0.1,
        'lmbda':     0.8,
        'e_rand':    1.0,
        'color':     'green'
    }
    td_lambda_offline_10 = {
        'method':    'td-lambda-offline',
        'nb_states': 11,
        'stepsize':  0.1,
        'lmbda':     0.8,
        'e_rand':    1.0,
        'color':     'red'
    }
    tests = [mc_full,
        td_lambda_offline_1000,
        td_lambda_offline_100,
        td_lambda_offline_10]
    #tests = [td_offline]

    for test in tests:
        np.random.seed(0)
        print(test['method'])

        test['RMSE'], test['final_V'], test['final_Q'] = test_run(
            nb_episodes=nb_episodes,
            method=test['method'],
            nb_states=test['nb_states'],
            step_size=test['stepsize'],
            lmbda=test['lmbda'],
            e_rand=test['e_rand'])


    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    X = np.array(range(-500, 501))
    ax1.plot(LinearEnv.TRUE_VALUES, label='True V', color='black')
    
    for test in tests:

        label = test['method']

        arr_Vp = np.mean(test['final_Q'], axis=1)


        ax1.plot(arr_Vp, label=label, color=test['color'], alpha=0.3)

          # average between two actions (should be the same as final_V above)
        # final_V_from_Q = np.mean(test['final_Q'][i], axis=1)
        # ax1.plot(final_V_from_Q[1:-1], label=label, color=test['color'], alpha=1.0)        

        # ax2.plot(test['RMSE'][i], label=label, color=test['color'], alpha=0.3)

        # label = None

        ax2.plot(test['RMSE'], label=label, color=test['color'])


    plt.legend()

    plt.grid()
    plt.show()


def test_single():
    rmse, arr_V, arr_Q = \
        test_run(100, 'td-lambda-offline',
            step_size=0.1, lmbda=0.8, e_rand=1.0)

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
    #ax.plot(X, arr_Q[:,0], color='green', label='Q_left')
    #ax.plot(X, arr_Q[:,1], color='red', label='Q_right')
    ax.grid()

    
    ax = fig.add_subplot(122)
    ax.plot(rmse)

    plt.show()

if __name__ == '__main__':
    # test_single()
    test_multi()
    
