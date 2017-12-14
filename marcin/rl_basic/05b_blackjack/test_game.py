import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import pdb

from blackjack import BlackjackEnv
from agent_vq import AgentVQ

def test_run(nb_episodes, method, step_size, nb_steps=None, lmbda=None):

    PLAYER_SUM_MIN = 12   # BlackjackEnv guarantees this
    PLAYER_SUM_MAX = 31   # 21 + draw 10
    DEALER_CARD_MIN = 1   # ace
    DEALER_CARD_MAX = 10

    state_space = []
    for has_ace in (1, 0):
        for player_sum in range(PLAYER_SUM_MIN, PLAYER_SUM_MAX+1):
            for dealer_card in range(DEALER_CARD_MIN, DEALER_CARD_MAX+1):
                state_space.append( (has_ace, player_sum, dealer_card) )

    action_space = [0, 1]  # stick, draw


    env = BlackjackEnv()
    agent = AgentVQ(state_space=state_space,
                action_space=action_space,
                step_size=step_size,
                nb_steps=nb_steps,
                lmbda=lmbda)  

    for e in range(nb_episodes):

        obs = env.reset()
        agent.reset()
        agent.append_trajectory(t_step=0,
                                prev_action=None,
                                observation=obs,
                                reward=None,
                                done=None)

        # if e % 10 == 0:
        #     agent._step_size *= 0.95
        #     print(agent._step_size)

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
                if method == 'mc-offline':
                    agent.eval_mc_offline()
                elif method == 'td-offline':
                    agent.eval_td_offline()
                elif method == 'td-lambda-offline':
                    agent.eval_td_lambda_offline()
                break




    return agent.V, agent.Q



def test_single():
    nb_episodes = 100000

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
    #tests = [td_offline, mc_offline, td_lambda_offline]
    tests = [mc_offline]

    for test in tests:
        np.random.seed(0)
        print(test['method'])
        test['V_dict'], test['Q_dict'] = test_run(
            nb_episodes=nb_episodes, method=test['method'],
            step_size=test['stepsize'], nb_steps=test['nb_steps'],
            lmbda=test['lmbda'])











    # convert to 2d arrays
    V = tests[0]['V_dict']


    # no ace state-values
    player_points = list(range(12, 22))
    dealer_card = list(range(1, 11))
    X, Y = np.meshgrid(dealer_card, player_points)
    Z = np.zeros([len(player_points), len(dealer_card)])

    for dc in dealer_card:
        for pp in player_points:
            val = V[(0, pp, dc)]
            Z[player_points.index(pp), dealer_card.index(dc)] = val

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z)
    plt.show()









    return




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
    test_single()
    
