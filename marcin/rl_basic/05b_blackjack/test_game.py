import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import pdb

from blackjack import BlackjackEnv
from agent_vq import AgentVQ
from logger import Logger

PLAYER_SUM_MIN = 12   # BlackjackEnv guarantees this
PLAYER_SUM_MAX = 31   # 21 + draw 10
DEALER_CARD_MIN = 1   # ace
DEALER_CARD_MAX = 10

class RefData:
    def __init__(self, filename, ax=None):

        self.ax = ax  # axis to draw plot

        dataset = np.load(filename)

        self.ref_no_ace_hold = dataset[:,:,0,1]
        self.ref_no_ace_draw = dataset[:,:,0,0]
        self.ref_ace_hold = dataset[:,:,1,1]
        self.ref_ace_draw = dataset[:,:,1,0]

        self.log_t = []
        self.log_rmse_no_ace_hold = []
        self.log_rmse_no_ace_draw = []
        self.log_rmse_ace_hold = []
        self.log_rmse_ace_draw = []
        self.log_rsme_total = []

    def calc_RMSE(self, episode, V, Q, b=False):
        Q_no_ace_hold = np.zeros([10, 10])
        Q_no_ace_draw = np.zeros([10, 10])
        Q_ace_hold = np.zeros([10, 10])
        Q_ace_draw = np.zeros([10, 10])

        for player_sum in range(PLAYER_SUM_MIN, 21+1):
            for dealer_card in range(DEALER_CARD_MIN, DEALER_CARD_MAX+1):
                Q_no_ace_hold[player_sum-12, dealer_card-1] = \
                    Q[(0, player_sum, dealer_card), 0]
                Q_no_ace_draw[player_sum-12, dealer_card-1] = \
                    Q[(0, player_sum, dealer_card), 1]
                Q_ace_hold[player_sum-12, dealer_card-1] = \
                    Q[(1, player_sum, dealer_card), 0]
                Q_ace_draw[player_sum-12, dealer_card-1] = \
                    Q[(1, player_sum, dealer_card), 1]

        self.log_t.append(episode)

        size = Q_no_ace_hold.size

        sum1 = np.sum(np.power(self.ref_no_ace_hold - Q_no_ace_hold, 2))
        rsme = np.sqrt(sum1 / size)
        self.log_rmse_no_ace_hold.append(rsme)

        sum2 = np.sum(np.power(self.ref_no_ace_draw - Q_no_ace_draw, 2))
        rsme = np.sqrt(sum2 / size)
        self.log_rmse_no_ace_draw.append(rsme)

        sum3 = np.sum(np.power(self.ref_ace_hold - Q_ace_hold, 2))
        rsme = np.sqrt(sum3 / size)
        self.log_rmse_ace_hold.append(rsme)

        sum4 = np.sum(np.power(self.ref_ace_draw - Q_ace_draw, 2))
        rsme = np.sqrt(sum4 / size)
        self.log_rmse_ace_draw.append(rsme)

        rsme_total = np.sqrt( (sum1 + sum2 + sum3 + sum4) / size * 4 )
        self.log_rsme_total.append(rsme_total)

        rsme_arr = np.power(self.ref_no_ace_hold - Q_no_ace_hold, 2)
        return Q_no_ace_hold, rsme_arr

    def plot(self):
        if self.ax is not None:
            self.ax.clear()
            self.ax.plot(self.log_t, self.log_rmse_no_ace_hold, 
                color='green', linestyle='solid')
            self.ax.plot(self.log_t, self.log_rmse_no_ace_draw, 
                color='red', linestyle='solid')

            self.ax.plot(self.log_t, self.log_rmse_ace_hold, 
                color='green', linestyle='dashed')
            self.ax.plot(self.log_t, self.log_rmse_ace_draw, 
                color='red', linestyle='dashed')



def test_run(nb_episodes, method, step_size,
    nb_steps=None, lmbda=None, ax=None, ref=None, log=None):

    # States are encoded as:
    # (   HAS_ACE   ,   PLAYER SUM   ,   DEALER CARD   )

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
            print('episode:', e, '/', nb_episodes, '   ')

            if log is not None:
                log.log(e, agent.V, agent.Q, agent.Q_num)

            # if ax is not None:
            #    plot_3d_points(ax, agent.V, agent.Q, label='rt', color='purple')
            #    plt.pause(0.001)
            #    pass

            """
            if ref is not None:
                qqq, rmse = ref.calc_RMSE(e, agent.V, agent.Q, True)
                ref.plot()
                # print('rmse: ', rmse)

                player_points = list(range(12, 22))
                dealer_card = list(range(1, 11))
                X, Y = np.meshgrid(dealer_card, player_points)
                # Z_Q0 = np.zeros([len(player_points), len(dealer_card)])  # stick
                # Z_Q1 = np.zeros([len(player_points), len(dealer_card)])  # draw

                # for dc in dealer_card:
                #     for pp in player_points:
                #         val = V[(0, pp, dc)]
                #         Z_V[player_points.index(pp), dealer_card.index(dc)] = val

                #         val = Q[(0, pp, dc), 0]
                #         Z_Q0[player_points.index(pp), dealer_card.index(dc)] = val

                #         val = Q[(0, pp, dc), 1]
                #         Z_Q1[player_points.index(pp), dealer_card.index(dc)] = val

                ax.clear()
                ax.plot_wireframe(X, Y, ref.ref_no_ace_hold, label='hold', color='blue')
                ax.plot_wireframe(X, Y, qqq, label='q', color='green')
                ax.plot_wireframe(X, Y, rmse-1, label='rsme', color='red')

                ax.set_xlabel('X = dealer card')
                ax.set_ylabel('Y = player points')
                ax.set_zlabel('Z = value')

                plt.pause(0.001)
            """






        # obs = env.reset()

        obs = env.reset_exploring_starts()




        agent.reset_exploring_starts()
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
        'stepsize':  0.01,
        'nb_steps':  None,
        'lmbda':     0.0,
        'color':     'orange'
    }
    #tests = [td_offline, mc_offline, td_lambda_offline]
    #tests = [td_lambda_offline]
    tests = [td_lambda_offline]

    # plt.ion()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111)

    # ref_data = RefData('reference.npy', ax=None)

    

    for test in tests:
        logger = Logger(reference_data_filename='reference.npy')
        np.random.seed(0)
        print(' =================   ', test['method'], '   ====================== ')
        test['V_dict'], test['Q_dict'] = test_run(
            nb_episodes=nb_episodes, method=test['method'],
            step_size=test['stepsize'], nb_steps=test['nb_steps'],
            lmbda=test['lmbda'], ax=None, ref=None, log=logger)

        logger.save(test['method']+'.log')

    

    # for test in tests:
    #     # convert to 2d arrays
    #     V = test['V_dict']
    #     Q = test['Q_dict']
    #     plot_3d_points(ax, V, Q, label=test['method'], color=test['color'])

    # plt.ioff()
    # plt.show()






def plot_3d_wireframe(ax, V, Q, label, color):
    ax.clear()

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

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def plot_3d_points(ax, V, Q, label, color):
    ax.clear()

    # no ace state-values
    dealer_card = list(range(1, 11))
    player_points = list(range(12, 22))
    X, Y = np.meshgrid(dealer_card, player_points)
    Z_V = np.zeros([len(player_points), len(dealer_card)])
    Z_Q0 = np.zeros([len(player_points), len(dealer_card)])  # stick
    Z_Q1 = np.zeros([len(player_points), len(dealer_card)])  # draw

    X = player_points
    Y_hold = []
    Y_draw = []

    for dc in dealer_card:

        Y_hold.clear()
        Y_draw.clear()

        for pp in player_points:     
            Y_hold.append(Q[(0, pp, dc), 0])
            Y_draw.append(Q[(0, pp, dc), 1])


        ax.plot(X, Y_hold, zs=dc, zdir='x', color='green')
        ax.plot(X, Y_draw, zs=dc, zdir='x', color='red')

    ax.set_xlim(1, 11)
    ax.set_ylim(12, 22)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')




if __name__ == '__main__':
    test_single()
    
