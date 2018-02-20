import numpy as np
import matplotlib.pyplot as plt
import pdb

from linear_pg import LinearEnvPG
from agent_pg import AgentPG

def test_run(nb_episodes, world_size, method,
            step_size, lmbda=None):
    env = LinearEnvPG (world_size)  # 2x terminal states will be added internally
    agent = AgentPG(world_size=world_size+2,    # add 2x terminal states
                action_space=2,
                step_size=step_size,
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

    return RMSE, agent.V, agent.Q

def multi_run(nb_runs, nb_episodes, world_size, method,
            step_size, lmbda=None):
    multi_RMSE = []
    multi_final_V = []
    multi_final_Q = []

    for run in range(nb_runs):
        RMSE, final_V, final_Q = test_run(nb_episodes, world_size, method, 
                                 step_size, lmbda)

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
    world_size = 19

    # Experiments tuned for world size 19
    td_offline = {
        'method':    'td-offline',
        'stepsize':  0.15,
        'lmbda':     None,
        'color':     'blue'
    }
    mc_offline = {
        'method':    'mc-offline',
        'stepsize':  0.01,
        'lmbda':     1.0,     # Monte-Carlo
        'color':     'red'
    }
    td_lambda_offline = {
        'method':    'td-lambda-offline',
        'stepsize':  0.15,
        'lmbda':     0.3,
        'color':     'orange'
    }
    tests = [td_offline, mc_offline, td_lambda_offline]
    # tests = [mc_offline]

    for test in tests:
        np.random.seed(0)
        print(test['method'])
        # test_A_RMSE, test_A_final_V = multi_run(
        #     nb_runs=10, nb_episodes=100, world_size=world_size, 
        #     method='N-step-offline', step_size=0.2, n_steps=1)
        test['RMSE'], test['final_V'], test['final_Q'] = multi_run(
            nb_runs=nb_runs, nb_episodes=nb_episodes, world_size=world_size, 
            method=test['method'], step_size=test['stepsize'], lmbda=test['lmbda'])


    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(LinearEnvPG.GROUND_TRUTH[world_size][1:-1], label='True V', color='black')
    
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
    
