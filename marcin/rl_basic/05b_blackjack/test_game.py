import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import time
import pickle
import pdb

from blackjack import BlackjackEnv
from agent import Agent
from agent_vq import AgentOld
from logger import DataLogger, DataReference

PLAYER_SUM_MIN = 12   # BlackjackEnv guarantees this
PLAYER_SUM_MAX = 31   # 21 + draw 10
DEALER_CARD_MIN = 1   # ace
DEALER_CARD_MAX = 10


class ExpParams:
    def __init__(self, nb_episodes, agent, expl_starts,
                 method, step_size, lmbda, e_greed):
        self.nb_episodes = nb_episodes
        self.agent = agent
        self.expl_starts = expl_starts
        self.method = method
        self.step_size = step_size
        self.lmbda = lmbda
        self.e_greed = e_greed
        
    def __hash__(self):
        return hash((self.nb_episodes, self.agent,
                     self.expl_starts, self.method, 
                     self.step_size, self.lmbda, self.e_greed))

    def __eq__(self, other):
        return (self.nb_episodes, self.agent,
                self.expl_starts, self.method, 
                self.step_size, self.lmbda, self.e_greed) == \
            (other.nb_episodes, other.agent,
             other.expl_starts, other.method, 
             other.step_size, other.lmbda, other.e_greed)

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not(self == other)
        
class ExpDesc:
    def __init__(self, color, redo):
        self.color = color
        self.redo = redo

class Experiment:
    def __init__(self, nb_episodes, agent, expl_starts,
                 method, step_size, lmbda, e_greed, color, redo):
        self.params = ExpParams(nb_episodes, agent, expl_starts,
                                method, step_size, lmbda, e_greed)
        self.desc = ExpDesc(color, redo)
        self.data_logger = None

    def __str__(self):
        data_logger_present = self.data_logger is not None
        return 'Params: ep={0} ag={1} es={2} m={3} step={4} lmbda={5} e_greed={6}; Data={7}'.format(
            self.params.nb_episodes,
            self.params.agent,
            self.params.expl_starts,
            self.params.method,
            self.params.step_size,
            self.params.lmbda, 
            self.params.e_greed,
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

    def delete_redo(self, exp_list):
        for exp in exp_list:
            if exp.desc.redo:
                if exp.params in self.exp_dict:
                    del self.exp_dict[exp.params]


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
    if experiment.params.agent == 'old':
        agent = AgentOld(state_space=state_space,
                    action_space=action_space,
                    step_size=experiment.params.step_size,
                    lmbda=experiment.params.lmbda)
    elif experiment.params.agent == 'new':
        agent = Agent(state_space=state_space,
                    action_space=action_space,
                    step_size=experiment.params.step_size,
                    lmbda=experiment.params.lmbda)
    else:
        raise ValueError('Agent type not recognized')

    for e in range(experiment.params.nb_episodes):
        if e % 1000 == 0:
            print('episode:', e, '/', experiment.params.nb_episodes, '   ')

            if experiment.data_logger is not None:
                experiment.data_logger.log(e, agent.V, agent.Q, agent.Q_num)


        # obs = env.reset()
        if experiment.params.expl_starts:
            obs = env.reset_exploring_starts()
            agent.reset_exploring_starts()
        else:
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

            method = experiment.params.method
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


    #
    #   MC-FULL (ES)  -  REFERENCE
    #
    # exp_lm = Experiment(
    #     nb_episodes=nb_episodes*1, agent='old',
    #     expl_starts=True, method='mc-full',
    #     step_size=0.005, lmbda=1.0, e_greed=0.0,
    #     color='gray', redo=False)
    # exp_list.append(exp_lm)

    # exp_lm = Experiment(
    #     nb_episodes=nb_episodes*1, agent='new',
    #     expl_starts=True, method='mc-full',
    #     step_size=0.005, lmbda=1.0, e_greed=0.0,
    #     color='green', redo=True)
    # exp_list.append(exp_lm)


    #
    #   ES - MC
    #
    # exp_lm = Experiment(
    #     nb_episodes=nb_episodes*1, agent='old',
    #     expl_starts=True, method='mc-offline',
    #     step_size=0.005, lmbda=None, e_greed=0.0,
    #     color='orange', redo=False)
    # exp_list.append(exp_lm)
    # exp_lm = Experiment(
    #     nb_episodes=nb_episodes*1, agent='new',
    #     expl_starts=True, method='mc-offline',
    #     step_size=0.005, lmbda=None, e_greed=0.0,
    #     color='red', redo=True)
    # exp_list.append(exp_lm)



    #
    #   ES - TD
    #
    # exp_lm = Experiment(
    #     nb_episodes=nb_episodes*1, agent='old',
    #     expl_starts=True, method='td-offline',
    #     step_size=0.005, lmbda=None, e_greed=0.0,
    #     color='orange', redo=False)
    # exp_list.append(exp_lm)
    # exp_lm = Experiment(
    #     nb_episodes=nb_episodes*1, agent='new',
    #     expl_starts=True, method='td-offline',
    #     step_size=0.005, lmbda=None, e_greed=0.0,
    #     color='blue', redo=True)
    # exp_list.append(exp_lm)


    #
    #   ES - TD(lmbda)
    #
    exp_lm = Experiment(
        nb_episodes=nb_episodes*1, agent='old',
        expl_starts=True, method='td-lambda-offline',
        step_size=0.005, lmbda=0.8, e_greed=0.0,
        color='orange', redo=False)
    exp_list.append(exp_lm)
    exp_lm = Experiment(
        nb_episodes=nb_episodes*1, agent='new',
        expl_starts=True, method='td-lambda-offline',
        step_size=0.005, lmbda=0.8, e_greed=0.0,
        color='blue', redo=True)
    exp_list.append(exp_lm)


    # exp_lm = Experiment(
    #     nb_episodes=nb_episodes*1,
    #     expl_starts=False, method='td-lambda-offline',
    #     step_size=0.1, lmbda=1.0, e_greed=0.1, 
    #     color='blue', redo=False)
    # exp_list.append(exp_lm)

    # exp_lm = Experiment(
    #     nb_episodes=nb_episodes*1,
    #     expl_starts=False, method='td-lambda-offline',
    #     step_size=0.005, lmbda=1.0, e_greed=0.1,
    #     color='blue', redo=False)
    # exp_list.append(exp_lm)

    # exp_lm = Experiment(
    #     nb_episodes=nb_episodes*3,
    #     expl_starts=False, method='td-lambda-offline',
    #     step_size=0.005, lmbda=1.0, e_greed=0.1,
    #     color='blue', redo=False)
    # exp_list.append(exp_lm)


    exp_db.delete_redo(exp_list)
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

            time_start = time.time()
            test_run(exp)
            time_total = time.time() - time_start
            print('Time Total: ', time_total)

            exp.data_logger.prep_to_save()
        else:
            # Do nothing, experiment results were loaded from file
            pass

    exp_db.put_to_db(exp_list)
    exp_db.save_to_file()


    #
    #   Plotting
    #
    
    #
    #   Reference Policy
    #

    # fig_ref = plt.figure('Reference Policy')
    # ax_ref_no_ace = fig_ref.add_subplot(121)
    # ax_ref_ace = fig_ref.add_subplot(122)
    # plot_policy(ax_ref_no_ace,
    #     data_ref.Q_no_ace_hold,
    #     data_ref.Q_no_ace_draw)
    # plot_policy(ax_ref_ace,
    #     data_ref.Q_ace_hold,
    #     data_ref.Q_ace_draw)

    #
    #   RMSE
    #

    fig_rmse = plt.figure('RSME')
    ax_rmse = fig_rmse.add_subplot(111)
    for exp in exp_list:
        exp.data_logger.process_data(data_ref)
        plot_rsme(ax_rmse, exp, data_ref)
    
    #
    #   Wireframe & Policy
    #

    for exp in exp_list:

        fig = plt.figure(exp.__str__())
        ax_wireframe_no_ace = fig.add_subplot(231, projection='3d', title='No Ace')
        ax_policy_no_ace = fig.add_subplot(232)
        ax_rmse_no_ace = fig.add_subplot(233)
        ax_wireframe_ace = fig.add_subplot(234, projection='3d', title='Ace')
        ax_policy_ace = fig.add_subplot(235)
        ax_rmse_ace = fig.add_subplot(236)


        # Wireframe Plot
        plot_experiment(ax_wireframe_no_ace, ax_wireframe_ace, exp, data_ref)

        # Policy Plot
        #plot_policy_log(exp, 'Policy for ' + exp.__str__())

        plot_policy(ax_policy_no_ace,
            exp.data_logger.Q_no_ace_hold[-1],
            exp.data_logger.Q_no_ace_draw[-1])
        plot_policy(ax_policy_ace,
            exp.data_logger.Q_ace_hold[-1],
            exp.data_logger.Q_ace_draw[-1])

        plot_rsme(ax_rmse_no_ace, exp, data_ref)

        

    plt.show()

def plot_policy(ax, Q_hold, Q_draw):
    policy = (Q_draw > Q_hold).astype(int)

    ax.imshow(policy, origin='lower', extent=[0.5,10.5,11.5,21.5], interpolation='none')
    ax.set_xticks(np.arange(0.5, 10.5, 1))
    ax.set_yticks(np.arange(11.5, 21.5, 1))
    ax.grid()





def plot_rsme(ax, exp, ref):
    log = exp.data_logger
    color = exp.desc.color

    ax.plot(log.t, log.rmse_no_ace_hold, color=color, linestyle='--')
    ax.plot(log.t, log.rmse_no_ace_draw, color=color, linestyle='-')
    ax.plot(log.t, log.rmse_ace_hold, color=color, linestyle=':')
    ax.plot(log.t, log.rmse_ace_draw, color=color, linestyle='-.')

    ax.plot(log.t, log.rmse_total, color=color, linestyle='-')

    ax.plot(log.t, np.zeros_like(log.t), color='black')

    ax.grid()




def plot_3d_wireframe(ax, Z, label, color):
    # ax.clear()

    dealer_card = list(range(1, 11))
    player_points = list(range(12, 22))
    X, Y = np.meshgrid(dealer_card, player_points)
    ax.plot_wireframe(X, Y, Z, label=label, color=color)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def plot_experiment(ax_no_ace, ax_ace, exp, ref):
    log = exp.data_logger
    
    plot_3d_wireframe(ax_no_ace, ref.Q_no_ace_hold, 'hold', (0.5, 0.7, 0.5, 1.0))
    plot_3d_wireframe(ax_no_ace, ref.Q_no_ace_draw, 'draw', (0.7, 0.5, 0.5, 1.0))
    plot_3d_wireframe(ax_no_ace, log.Q_no_ace_hold[-1], 'hold', 'green')
    plot_3d_wireframe(ax_no_ace, log.Q_no_ace_draw[-1], 'draw', 'red')
    ax_no_ace.set_zlim(-1, 1)

    plot_3d_wireframe(ax_ace, ref.Q_ace_hold, 'hold', (0.5, 0.7, 0.5, 1.0))
    plot_3d_wireframe(ax_ace, ref.Q_ace_draw, 'draw', (0.7, 0.5, 0.5, 1.0))
    plot_3d_wireframe(ax_ace, log.Q_ace_hold[-1], 'hold', 'green')
    plot_3d_wireframe(ax_ace, log.Q_ace_draw[-1], 'draw', 'red')
    ax_ace.set_zlim(-1, 1)






def main2():
    print('hoho')

    Q = OffsetQ()
    Q[(1, 12, 1), 1] = 30

    print(Q._data)


if __name__ == '__main__':
    main()
    
