import numpy as np
import matplotlib.pyplot as plt
from gridworld import GridworldEnv



class GridworldAgent:
    def __init__(self, size_x, size_y):
        self.size_x = size_x
        self.size_y = size_y

        # [position x, position y, direction]
        # directions: 0 = N; 1 = E; 2 = S; 3 = W
        self._Q = np.zeros([size_x, size_y, 4])

    def policy_rand(self, state_x, state_y):
        return np.random.randint(0, 4)



size_x = 6
size_y = 6

env = GridworldEnv(size_x, size_y)
agent = GridworldAgent(size_x, size_y)

fig = plt.figure('Value Iteration')
ax0 = fig.add_subplot(111)

QQ = np.zeros([size_x, size_y, 4])
QQ[0, 0, 0] = 1
QQ[0, 0, 1] = 0.25
QQ[0, 0, 2] = 0.5

env.plot_world(ax0, 'k=0', Q=QQ)

plt.show()