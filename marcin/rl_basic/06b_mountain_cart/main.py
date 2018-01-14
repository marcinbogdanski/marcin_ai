import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pdb

from mountain_car import MountainCarEnv
from agent import Agent


def test_run(nb_episodes, approximator, step_size, e_rand):


    action_space = [-1, 0, 1]  # move left, do nothing, move right

    env = MountainCarEnv()
    agent = Agent(action_space=action_space,
                approximator=approximator,
                step_size=step_size,
                e_rand=e_rand)
    

    # RMSE = []    # root mean-squared error
    e = 0
    for e in range(nb_episodes):
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

            agent.eval_td_online()
            
            if done:
                break

        arr_Q = agent.Q._states  # [pos, vel, act]
        arr_V = np.max(arr_Q, axis=2)
        # 
        # for i in range(len(arr_V)):
        #     arr_Q[i, 0] = agent.Q[i, 0]
        #     arr_Q[i, 1] = agent.Q[i, 1]
        # arr_Vp = np.mean(arr_Q, axis=1)

        # rmse = np.sqrt(np.sum(np.power(
        #     LinearEnv.TRUE_VALUES - arr_Vp, 2)) / len(LinearEnv.TRUE_VALUES))
        # RMSE.append(rmse)
    
    return arr_V, arr_Q




def test_multi():
    nb_episodes = 2000

    # Experiments tuned for world size 19
    mc_full = {
        'name':      'agg mc-full',
        'agent':     'aggregate',
        'method':    'mc-full',
        'nb_states': 1001,
        'stepsize':  None,
        'lmbda':     None,
        'e_rand':    1.0,
        'color':     'gray'
    }
    td_lambda_offline_1000 = {
        'name':      'agg td-lambda-offline 1001',
        'agent':     'aggregate',
        'method':    'td-lambda-offline',
        'nb_states': 1001,
        'stepsize':  0.1,
        'lmbda':     0.8,
        'e_rand':    1.0,
        'color':     'orange'
    }
    td_lambda_offline_100 = {
        'name':      'agg td-lambda-offline 101',
        'agent':     'aggregate',
        'method':    'td-lambda-offline',
        'nb_states': 101,
        'stepsize':  0.1,
        'lmbda':     0.8,
        'e_rand':    1.0,
        'color':     'green'
    }
    td_lambda_offline_10 = {
        'name':      'agg td-lambda-offline 11',
        'agent':     'aggregate',
        'method':    'td-lambda-offline',
        'nb_states': 11,
        'stepsize':  0.1,
        'lmbda':     0.8,
        'e_rand':    1.0,
        'color':     'red'
    }
    linear_mc = {
        'name':      'lin mc-offline',
        'agent':     'linear',
        'method':    'mc-offline',
        'nb_states': None,
        'stepsize':  0.0015,
        'lmbda':     None,
        'e_rand':    1.0,
        'color':     'blue'
    }
    linear_td = {
        'name':      'lin td-offline',
        'agent':     'linear',
        'method':    'td-offline',
        'nb_states': None,
        'stepsize':  0.015,
        'lmbda':     None,
        'e_rand':    1.0,
        'color':     'purple'
    }
    tests = [
        # mc_full,
        # td_lambda_offline_1000,
        # td_lambda_offline_100,
        # td_lambda_offline_10,
        linear_mc,
        linear_td]
    #tests = [td_offline]

    for test in tests:
        np.random.seed(0)
        print(test['method'])

        test['RMSE'], test['final_V'], test['final_Q'] = test_run(
            nb_episodes=nb_episodes,
            agent_type=test['agent'],
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

        label = test['name']

        arr_Vp = np.mean(test['final_Q'], axis=1)
        arr_Q = test['final_Q']


        ax1.plot(arr_Vp, label=label, color=test['color'], alpha=0.3)

        #ax1.plot(arr_Q[:,0], label=label, color='green', alpha=0.3)
        #ax1.plot(arr_Q[:,1], label=label, color='red', alpha=0.3)

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
    nb_episodes = 100
    arr_V, arr_Q = \
        test_run(nb_episodes=nb_episodes,
            approximator=None, step_size=0.1, e_rand=0.0)

    
    # X = np.array(range(-500, 501))

    # arr_V = np.zeros(1001)
    # arr_Q = np.zeros([1001, 2])
    # for i in range(len(arr_V)):
    #     arr_V[i] = agent_V[i]
    #     arr_Q[i, 0] = agent_Q[i, 0]
    #     arr_Q[i, 1] = agent_Q[i, 1]
    # arr_Vp = np.mean(arr_Q, axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    pos = list(range(0, 50))
    vel = list(range(0, 50))
    X, Y = np.meshgrid(pos, vel)
    ax.plot_wireframe(X, Y, -arr_V)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # ax.plot(X, LinearEnv.TRUE_VALUES, color='gray', label='V-ref')
    # ax.plot(X, arr_Vp, color='orange', label='V')
    # ax.grid()

    
    # ax = fig.add_subplot(122)
    # ax.plot(rmse)

    plt.show()

def test_cart():

    np.random.seed(0)

    env = MountainCarEnv()

    x = []
    v = []
    a = []

    obs = env.reset()
    print('obs', obs)
    x.append(obs[0])
    v.append(obs[1])

    for i in range(500):

        action = 1 # np.random.choice([-1, 0, 1])
        a.append(action)
        obs, reward, done = env.step(action)
        print('obs, rew, done', obs, reward, done)
        x.append(obs[0])
        v.append(obs[1])

        if done:
            break


    fig = plt.figure()


    ax1 = fig.add_subplot(121)
    X = np.linspace(-1.2, 0.5, 100)
    ax1.plot(X, -0.0025*np.cos(3*X))

    ax2 = fig.add_subplot(122)
    ax2.plot(a, color='red')
    ax2.plot(x, color='blue')
    ax2.plot(v, color='green')

    plt.show()

def test_agg():
    from agent import AggregateApproximator
    
    aa = AggregateApproximator(step_size = 0.01)

    pos_num = np.zeros([50])
    vel_num = np.zeros([50])

    val, pi, vi = aa._to_idx((0.5, 0.07), 0)
    print('val, pi, vi', val, pi, vi)
    val, pi, vi = aa._to_idx((0.0, 0.07), 0)
    print('val, pi, vi', val, pi, vi)
    val, pi, vi = aa._to_idx((-1.2, 0.07), 0)
    print('val, pi, vi', val, pi, vi)
    val, pi, vi = aa._to_idx((0.5, -0.07), 0)
    print('val, pi, vi', val, pi, vi)
    val, pi, vi = aa._to_idx((0.0, -0.07), 0)
    print('val, pi, vi', val, pi, vi)
    val, pi, vi = aa._to_idx((-1.2, -0.07), 0)
    print('val, pi, vi', val, pi, vi)

    for i in range(10000):
        rand_pos = np.random.uniform(-1.2, 0.5)
        rand_vel = np.random.uniform(-0.07, 0.07)

        pi, vi, ai = aa._to_idx((rand_pos, rand_vel), 0)

        pos_num[pi] += 1
        vel_num[vi] += 1

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(pos_num, color='blue')
    ax.plot(vel_num, color='green')

    plt.show()


if __name__ == '__main__':
    test_single()
    # test_multi()
    # test_cart()
    # test_agg()
    
