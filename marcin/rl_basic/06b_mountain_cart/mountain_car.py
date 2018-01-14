import numpy as np
import matplotlib.pyplot as plt
import pdb

class MountainCarEnv:

    def __init__(self):
        self.t_step = 0
        self._pos = 0
        self._vel = 0
        self.reset()

    def reset(self):
        self.t_step = 0
        self._pos = np.random.uniform(-0.6, -0.4)
        self._vel = 0.0
        self._done = False

        return (self._pos, self._vel)


    def step(self, action):
        assert self._done == False
        assert action in [-1, 0, 1]

        self.t_step += 1

        self._vel = self._vel + 0.001*action - 0.0025*np.cos(3*self._pos)
        self._vel = min(max(self._vel, -0.07), 0.07)

        self._pos = self._pos + self._vel
        self._pos = min(max(self._pos, -1.2), 0.5)

        if self._pos == -1.2:
            self._vel = 0.0

        if self._pos == 0.5:
            obs = (self._pos, self._vel)
            reward = 0
            self._done = True
            return (obs, reward, self._done)
        else:
            obs = (self._pos, self._vel)
            reward = -1
            return (obs, reward, self._done)


    def print_env(self):
        print('Env pos/vel: ', self._pos, self._vel)






