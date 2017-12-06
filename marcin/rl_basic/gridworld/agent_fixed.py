import numpy as np
import matplotlib.pyplot as plt
from gridworld import GridworldEnv
import pdb

class HistoryData:
    def __init__(self, t_step, obs, reward, done):
        self.t_step = t_step
        self.observation = obs
        self.reward = reward
        self.action = None
        self.done = done

class GridworldAgent:
    def __init__(self, size_x, size_y):
        self.size_x = size_x
        self.size_y = size_y

        # [position x, position y, direction]
        # directions: 0 = N; 1 = E; 2 = S; 3 = W
        self._Q = np.zeros([size_x, size_y, 4])

    def policy_rand(self, state_x, state_y):
        return np.random.randint(0, 4)

    def trajectory_append(self, t_step, prev_action, observation, reward, done):
        if len(self.trajectory) != 0:
            self.trajectory[-1].action = prev_action

        self.trajectory.append(
            HistoryData(t_step, obs, reward, done))

if __name__ == '__main__':


    size_x = 6
    size_y = 6

    env = GridworldEnv(size_x, size_y)
    env.make_start(0, 0)
    env.make_goal(5, 0)

    fig = plt.figure('Value Iteration')
    ax0 = fig.add_subplot(111)

    #for i in range(1000):
    #    pdb.set_trace()

    obs = env.reset()

    agent = GridworldAgent(size_x, size_y)


    total_ep = 1
    #while True:
    #    obs = env.reset()






    fig = plt.figure('Value Iteration')
    ax0 = fig.add_subplot(111)

    QQ = np.zeros([size_x, size_y, 4])
    QQ[0, 0, 0] = 1
    QQ[0, 0, 1] = 0.25
    QQ[0, 0, 2] = 0.5

    VV = np.zeros([size_x, size_y])

    trajectory = [ (0,0), (1,0), (2,0), (2,1), (1,1)]

    env.plot_world(ax0, 'k=0', 
        V=None,
        Q=None,
        trajectory=trajectory,
        plot_transitions=True,
        plot_rewards=True)

    plt.show()