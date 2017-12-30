import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import pickle
import pdb

from blackjack import BlackjackEnv
from agent_vq import AgentVQ
from logger import DataLogger, DataReference

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


class ExpParams:
    def __init__(self, nb_episodes, method, step_size, lmbda):
        self.nb_episodes = nb_episodes
        self.method = method
        self.step_size = step_size
        self.lmbda = lmbda
    def __hash__(self):
        return hash((self.nb_episodes, self.method, self.step_size, self.lmbda))

    def __eq__(self, other):
        return (self.nb_episodes, self.method, self.step_size, self.lmbda) == \
            (other.nb_episodes, other.method, other.step_size, other.lmbda)

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not(self == other)
        
class ExpDesc:
    def __init__(self, color):
        self.color = color

class Experiment:
    def __init__(self, nb_episodes, method, step_size, lmbda, color):
        self.params = ExpParams(nb_episodes, method, step_size, lmbda)
        self.desc = ExpDesc(color)
        self.data_logger = None

    def __str__(self):
        data_logger_present = self.data_logger is not None
        return 'Params: ep={0} m={1} step={2} lmbda={3}; Data={4}'.format(
            self.params.nb_episodes, self.params.method,
            self.params.step_size, self.params.lmbda, 
            data_logger_present)


class ExperimentsDB:
    def __init__(self, filename):
        self.filename = filename
        self.exp_dict = {}
        self.need_saving = False

    def load_from_file(self):
        try:
            with open(self.filename, 'rb') as f:
                self.exp_dict = pickle.load(f)
        except:
            pass

    def fill_from_db(self, exp_list):
        for exp in exp_list:
            if exp.params in self.exp_dict:
                exp.data_logger = self.exp_dict[exp.params].data_logger

    def put_to_db(self, exp_list):
        for exp in exp_list:
            if exp.params not in self.exp_dict:
                self.exp_dict[exp.params] = exp
                self.need_saving = True

    def save_to_file(self):
        if self.need_saving:
            with open(self.filename, 'wb') as f:
                pickle.dump(self.exp_dict, f)


# def test_run(nb_episodes, method, step_size,
#     lmbda=None, log=None):
def test_run(experiment):

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
                step_size=experiment.params.step_size,
                lmbda=experiment.params.lmbda)  

    for e in range(experiment.params.nb_episodes):
        if e % 1000 == 0:
            print('episode:', e, '/', experiment.params.nb_episodes, '   ')

            if experiment.data_logger is not None:
                experiment.data_logger.log(e, agent.V, agent.Q, agent.Q_num)


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

            method = experiment.params.method
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




def main():
    
    exp_db = ExperimentsDB('experiments.db')
    exp_db.load_from_file()

    print('Experimetns in file:')
    for params, exp in exp_db.exp_dict.items():
        print(exp)
    print()


    nb_episodes = 100000
    exp_list = []

    # exp_td = Experiment(nb_episodes, 'td-offline', 0.001, None, 'blue' )
    # exp_list.append(exp1)

    # exp_mc = Experiment(nb_episodes, 'mc-offline', 0.001, None, 'purple')
    # exp_list.append(exp_mc)

    exp_lm = Experiment(nb_episodes*10, 'td-lambda-offline', 0.001, 1.0, 'orange')
    exp_list.append(exp_lm)

    exp_lm = Experiment(nb_episodes*5, 'td-lambda-offline', 0.005, 1.0, 'orange')
    exp_list.append(exp_lm)

    # exp_lm = Experiment(nb_episodes*3, 'td-lambda-offline', 0.01, 1.0, 'blue')
    # exp_list.append(exp_lm)

    # exp_lm = Experiment(nb_episodes*2, 'td-lambda-offline', 0.1, 1.0, 'red')
    # exp_list.append(exp_lm)

    exp_db.fill_from_db(exp_list)


    print('All experiments:')
    for params, exp in exp_db.exp_dict.items():
        print(exp)
    print()

    
    
    print('Defined experiments:')
    for exp in exp_list:
        print(exp)

    
    data_ref = DataReference('reference.npy')

    for exp in exp_list:
        np.random.seed(0)
        print(' === Exp: ', exp)
        if exp.data_logger is None:
            exp.data_logger = DataLogger()
            test_run(exp)
            exp.data_logger.prep_to_save()
        else:
            # Do nothing, experiment results were loaded from file
            pass

    exp_db.put_to_db(exp_list)
    exp_db.save_to_file()



    fig = plt.figure()
    ax = fig.add_subplot(111)
    for exp in exp_list:
        exp.data_logger.process_data(data_ref)
        plot_rsme(exp, data_ref, ax)

    for exp in exp_list:
        plot_experiment(exp, data_ref)

    plt.show()



def plot_rsme(exp, ref, ax):
    log = exp.data_logger

    ax.plot(log.t, log.rmse_no_ace_hold, color='green', linestyle='-')
    ax.plot(log.t, log.rmse_no_ace_draw, color='red', linestyle='-')
    ax.plot(log.t, log.rmse_ace_hold, color='green', linestyle='--')
    ax.plot(log.t, log.rmse_ace_draw, color='red', linestyle='--')

    ax.plot(log.t, log.rmse_total, color='gray', linestyle='--')

    ax.plot(log.t, np.zeros_like(log.t), color='black')



def plot_error(exp, ref, ax):
    log = exp.data_logger
    color = exp.desc.color

    PLAYER_SUM = 12   # [12..21]
    DEALER_CARD = 10  # [1..10]

    x = log.t
    # player_sum == 12, dealer_card == 10
    y = log.Q_no_ace_hold[:,PLAYER_SUM-12,DEALER_CARD-1]
    y2 = [ref.Q_no_ace_hold[PLAYER_SUM-12,DEALER_CARD-1] for _ in log.t]

    if color == 'red':
        y += 0.001

    ax.plot(x, y, label=exp.__str__(), color=color)
    ax.plot(x, y2, label='Ref', color='black')

def plot_3d_wireframe(ax, Z, label, color):
    # ax.clear()

    dealer_card = list(range(1, 11))
    player_points = list(range(12, 22))
    X, Y = np.meshgrid(dealer_card, player_points)
    ax.plot_wireframe(X, Y, Z, label=label, color=color)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def plot_experiment(exp, ref):
    log = exp.data_logger
    fig = plt.figure(exp.__str__())
    ax = fig.add_subplot(121, projection='3d', title='No Ace')
    plot_3d_wireframe(ax, ref.Q_no_ace_hold, 'hold', (0.5, 0.7, 0.5, 1.0))
    plot_3d_wireframe(ax, ref.Q_no_ace_draw, 'draw', (0.7, 0.5, 0.5, 1.0))
    plot_3d_wireframe(ax, log.Q_no_ace_hold[-1], 'hold', 'green')
    plot_3d_wireframe(ax, log.Q_no_ace_draw[-1], 'draw', 'red')
    ax.set_zlim(-1, 1)

    ax = fig.add_subplot(122, projection='3d', title='Ace')
    plot_3d_wireframe(ax, ref.Q_ace_hold, 'hold', (0.5, 0.7, 0.5, 1.0))
    plot_3d_wireframe(ax, ref.Q_ace_draw, 'draw', (0.7, 0.5, 0.5, 1.0))
    plot_3d_wireframe(ax, log.Q_ace_hold[-1], 'hold', 'green')
    plot_3d_wireframe(ax, log.Q_ace_draw[-1], 'draw', 'red')
    ax.set_zlim(-1, 1)


if __name__ == '__main__':
    main()
    
