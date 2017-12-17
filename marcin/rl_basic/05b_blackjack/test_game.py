import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import pdb

from blackjack import BlackjackEnv
from agent_vq import AgentVQ

def test_run(nb_episodes, method, step_size, nb_steps=None, lmbda=None, ax=None):

    PLAYER_SUM_MIN = 12   # BlackjackEnv guarantees this
    PLAYER_SUM_MAX = 31   # 21 + draw 10
    DEALER_CARD_MIN = 1   # ace
    DEALER_CARD_MAX = 10

    state_space = []
    for has_ace in (1, 0):
        for player_sum in range(PLAYER_SUM_MIN, PLAYER_SUM_MAX+1):
            for dealer_card in range(DEALER_CARD_MIN, DEALER_CARD_MAX+1):
                state_space.append( (has_ace, player_sum, dealer_card) )

    # Blackjack will terminate without chaning state if player sticks
    # so we introduce special state to force correct evaluation of
    # state-value and action-value functions in terminal state (must equal zero)
    state_space.append('TERMINAL')

    action_space = [0, 1]  # stick, draw


    env = BlackjackEnv()
    agent = AgentVQ(state_space=state_space,
                action_space=action_space,
                step_size=step_size,
                nb_steps=nb_steps,
                lmbda=lmbda)  

    for e in range(nb_episodes):
        if e % 1000 == 0:
            print('episode:', e, '/', nb_episodes)

            if ax is not None:
                ax.clear()
                plot_3d(ax, agent.V, agent.Q, label='rt', color='purple')
                plt.pause(0.001)
                pass

        obs = env.reset()
        agent.reset()
        agent.append_trajectory(t_step=0,
                                prev_action=None,
                                observation=obs,
                                reward=None,
                                done=None)

        # if e % 100 == 0:
        #     agent._step_size *= 0.995
        #     print('                          ', agent._step_size)

        while True:

            action = agent.pick_action(obs)

            #   ---   time step rolls here   ---

            obs, reward, done = env.step(action)

            if done:
                # Force unique terminal state
                obs = 'TERMINAL'

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

                #agent.print_trajectory()
                #print('PLAYER:', env.player_hand)
                #print('DEALER:', env.dealer_hand)
                #pdb.set_trace()

                if method == 'mc-offline':
                    agent.eval_mc_offline()
                elif method == 'td-offline':
                    agent.eval_td_offline()
                elif method == 'td-lambda-offline':
                    agent.eval_td_lambda_offline()
                break




    return agent.V, agent.Q



def test_single():
    nb_episodes = 500000

    # Experiments tuned for world size 19
    td_offline = {
        'method':    'td-offline',
        'stepsize':  0.001,
        'nb_steps':  None,
        'lmbda':     None,
        'color':     'blue'
    }
    mc_offline = {
        'method':    'mc-offline',
        'stepsize':  0.001,
        'nb_steps':  None,
        'lmbda':     1.0,     # Monte-Carlo
        'color':     'purple'
    }
    td_lambda_offline = {
        'method':    'td-lambda-offline',
        'stepsize':  0.001,
        'nb_steps':  None,
        'lmbda':     0.5,
        'color':     'orange'
    }
    #tests = [td_offline, mc_offline, td_lambda_offline]
    #tests = [td_lambda_offline]
    tests = [td_lambda_offline]

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for test in tests:
        np.random.seed(0)
        print(' =================   ', test['method'], '   ====================== ')
        test['V_dict'], test['Q_dict'] = test_run(
            nb_episodes=nb_episodes, method=test['method'],
            step_size=test['stepsize'], nb_steps=test['nb_steps'],
            lmbda=test['lmbda'], ax=ax)



    

    for test in tests:

        # convert to 2d arrays
        V = test['V_dict']
        Q = test['Q_dict']

        ax.clear()
        plot_3d(ax, V, Q, label=test['method'], color=test['color'])

    plt.ioff()
    plt.show()


def plot_3d(ax, V, Q, label, color):

    # no ace state-values
    player_points = list(range(12, 22))
    dealer_card = list(range(1, 11))
    X, Y = np.meshgrid(dealer_card, player_points)
    Z_V = np.zeros([len(player_points), len(dealer_card)])
    Z_Q0 = np.zeros([len(player_points), len(dealer_card)])  # stick
    Z_Q1 = np.zeros([len(player_points), len(dealer_card)])  # draw

    for dc in dealer_card:
        for pp in player_points:
            val = V[(0, pp, dc)]
            Z_V[player_points.index(pp), dealer_card.index(dc)] = val

            val = Q[(0, pp, dc), 0]
            Z_Q0[player_points.index(pp), dealer_card.index(dc)] = val

            val = Q[(0, pp, dc), 1]
            Z_Q1[player_points.index(pp), dealer_card.index(dc)] = val

    #ax.plot_wireframe(X, Y, Z_V, label=label, color=color)
    ax.plot_wireframe(X, Y, Z_Q0, label='stick', color='green')
    ax.plot_wireframe(X, Y, Z_Q1, label='draw', color='red')

    #ax.plot(Y[0], Z_Q0[0,:], color='green')
    #ax.plot(Y[0], Z_Q1[0,:], color='red')


if __name__ == '__main__':
    test_single()
    
