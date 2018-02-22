import numpy as np
import matplotlib.pyplot as plt
import pdb

from linear_pg import LinearEnvPG
from agent_pg import AgentPG

def test_run(nb_episodes, world_size, expl_start, method,
            step_size, ax1=None, ax2=None, ax3=None):
    env = LinearEnvPG (world_size)  # 2x terminal states will be added internally
    agent = AgentPG(world_size=world_size+2,    # add 2x terminal states
                action_space=2,
                step_size=step_size)  

    RMSE = []    # root mean-squared error
    for e in range(nb_episodes):

        # pdb.set_trace()

        obs = env.reset(expl_start=expl_start)
        agent.reset(expl_start=expl_start)
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

        rms = np.sqrt(np.sum(np.power(
            LinearEnvPG.GROUND_TRUTH[world_size] - agent.V, 2)) / world_size)
        RMSE.append(rms)


        print('episode:', e)
        ax1.clear()
        if ax2 is not None: ax2.clear()
        if ax3 is not None: ax3.clear()
        ax1.plot(LinearEnvPG.GROUND_TRUTH[world_size][1:-1], label='True V', color='black')
        plot(ax1, ax2, ax3,
            label=method,
            color='red',
            agent_V=agent.V,
            agent_Q=agent.Q,
            agent_P=agent.get_pg_prob(),
            rmse=RMSE )
        plt.pause(0.1)

    return RMSE, agent.V, agent.Q, agent.get_pg_prob()


class Data:
    def __init__(self, label):
        self.label = label
        self.x = []
        self.y = []



def test_single():
    nb_runs = 5
    nb_episodes = 100
    world_size = 19

    # Experiments tuned for world size 19
    td_offline = {
        'expl_start': False,
        'method':    'td-offline',
        'stepsize':  0.1,
        'color':     'blue'
    }
    mc_offline = {
        'expl_start': False,
        'method':    'mc-offline',
        'stepsize':  0.1,
        'color':     'red'
    }
    tests = [td_offline, mc_offline]
    tests = [td_offline]

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = None
    ax3 = fig.add_subplot(122)

    for test in tests:
        np.random.seed(1)
        print(test['method'])
        test['RMSE'], test['final_V'], test['final_Q'], test['final_P'] = \
            test_run(
                nb_episodes=nb_episodes, world_size=world_size, 
                expl_start=test['expl_start'],
                method=test['method'], step_size=test['stepsize'],
                ax1=ax1, ax2=ax2, ax3=ax3)

    ax1.clear()
    if ax2 is not None: ax2.clear()
    if ax3 is not None: ax3.clear()
    ax1.plot(LinearEnvPG.GROUND_TRUTH[world_size][1:-1], label='True V', color='black')
    
    for test in tests:
        plot(ax1, ax2, ax3,
            label=test['method'],
            color=test['color'],
            agent_V=test['final_V'],
            agent_Q=test['final_Q'],
            agent_P=test['final_P'],
            rmse=test['RMSE'] )

    plt.legend()

    plt.grid()
    plt.show()


def plot(ax1, ax2, ax3, label, color, agent_V, agent_Q, agent_P, rmse):
    ax1.plot(agent_V[1:-1], label=label, color=color, alpha=1.0)

    # for LEFT action (0) here -> ----v
    ax1.plot(agent_Q[1:-1, 0], label=label, color=color, linestyle='--', alpha=1.0)

    # for RIGHT action (1) here -> ---v
    ax1.plot(agent_Q[1:-1, 1], label=label, color=color, linestyle=':', alpha=1.0)

    # average between two actions (should be the same as final_V above)
    #final_V_from_Q = np.mean(agent_Q, axis=1)
    #ax1.plot(final_V_from_Q[1:-1], label=label, color=color, alpha=1.0)        

    if ax2 is not None:
        ax2.plot(rmse, label=label, color=color, alpha=0.3)

    if ax3 is not None:
        ax3.plot(agent_P[1:-1, 0], label=label, color=color, linestyle='--')
        ax3.plot(agent_P[1:-1, 1], label=label, color=color, linestyle=':')
        ax3.set_ylim([0, 1])

if __name__ == '__main__':
    test_single()
    
