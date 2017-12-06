import numpy as np
import matplotlib.pyplot as plt
from gridworld import GridworldEnv
import pdb

class HistoryData:
    def __init__(self, t_step, observation, reward, done):
        self.t_step = t_step
        self.observation = observation
        self.reward = reward
        self.action = None
        self.done = done

    def __str__(self):
        return '{0}: {1}, {2} {3}   {4}'.format(
            self.t_step, self.observation, self.reward, self.done, self.action)

class GridworldAgent:
    def __init__(self, size_x, size_y):
        self.size_x = size_x
        self.size_y = size_y

        self.V = np.zeros([size_x, size_y])
        self.Q = np.zeros([self.size_x, self.size_y, 4])  # 0=N; 1=E; 2=S; 3=W

        self.step = 0.05   # step-size parameter - usually noted as alpha
        self.disc = 1.0    # discount factor - usually noted as gamma

        self.reset()

    def reset(self):
        self.trajectory = []

    def pick_action(self, observation):

        # ignore observation, return random move
        return np.random.randint(0, 4)
        # return 1

    def eval_td(self):
        # Shortcuts for more compact notation:
        V = self.V              # State values array, shape: [size_x, size_y]
        St = self.trajectory[-2].observation    # previous state tuple (x, y)
        St_1 = self.trajectory[-1].observation  # current state tuple (x, y)
        Rt_1 = self.trajectory[-1].reward       # current step reward

        V[St] = V[St] + self.step*(Rt_1 + self.disc*V[St_1] - V[St])

    def eval_mc(self):
        # Shortcuts for more compact notation:
        V = self.V             # State values array, shape: [size_x, size_y]

        # Do MC update once after episode completed
        if self.trajectory[-1].done:
            #pdb.set_trace()

            # Iterate all states in trajectory
            for t in range(0, len(self.trajectory)):


                St = self.trajectory[t].observation  # current state (x, y)
                Gt = 0            # return for current state

                # Iterate all states after this one
                for j in range(t+1, len(self.trajectory)):
                    Rj = self.trajectory[j].reward   # reward at time j
                    Gt += self.disc**j * Rj          # add with discount

                V[St] = V[St] + self.step*(Gt - V[St])



    def learn(self):
        pass

    def append_trajectory(self, t_step, prev_action, observation, reward, done):
        if len(self.trajectory) != 0:
            self.trajectory[-1].action = prev_action

        self.trajectory.append(
            HistoryData(t_step, obs, reward, done))

    def get_trajectory(self):
        result = []
        for element in self.trajectory:
            result.append( element.observation )
        return result

    def print_trajectory(self):
        print('Trajectory:')
        for element in self.trajectory:
            print(element)
        print('Total trajectory steps: {0}'.format(len(self.trajectory)))

if __name__ == '__main__':


    size_x = 4
    size_y = 4

    env = GridworldEnv(size_x, size_y)
    env.make_start(0, 0)
    env.make_goal(0, 3)
    env.make_goal(3, 0)

    agent = GridworldAgent(size_x, size_y)

    total_episodes = 1000
    for i in range(total_episodes):
        
        obs = env.reset()
        agent.reset()

        agent.append_trajectory(t_step=0,
                                prev_action=None,
                                observation=obs,
                                reward=None,
                                done=None)

        prev_action = agent.pick_action(obs)

        
        while True:

            #      --- time step rolls here ---
            #print('----  time step {0}  ----'.format(env.t_step))

            obs, reward, done = env.step(prev_action)

            #print(agent.V)

            agent.append_trajectory(t_step=env.t_step,
                                    prev_action=prev_action,
                                    observation=obs,
                                    reward=reward,
                                    done=done)

            agent.eval_td()  # learn from history

            #print(agent.V)

            if done:
                break
            else:
                prev_action = agent.pick_action(obs)



    total_ep = 1
    #while True:


    fig = plt.figure('Value Iteration')
    ax0 = fig.add_subplot(111)

    trajectory = agent.get_trajectory()

    env.plot_world(ax0, 'k=0', 
        V=agent.V,
        Q=None,
        trajectory=None,
        plot_transitions=False,
        plot_rewards=False)

    plt.show()