import numpy as np
import matplotlib.pyplot as plt
import pdb

class LinearEnv:
    """
    Allowed states are:
    [    0         1         2         3          4         5         6        ]
      terminal                      initial                        terminal
       state                         state                          state
             <--       <->       <->       <->         <->       -->
              -1        0         0         0           0         1
    """

    TRUE_VALUES = np.arange(-1001, 1001, 2) / 1000.0
    TRUE_VALUES *= 0.94  # correct for a fact that true values 
                         # are slightly tilted (ignore non-linearity)

    def __init__(self):
        self._max_left = 1          # last valid non-terminal state to the left
        self._max_right = 999       # last non-terminal state to the right
        self._start_state = 500
        self.reset()

    def reset(self):
        self.t_step = 0
        self._state = self._start_state
        self._done = False

        return self._state

    def step(self, action):
        if self._done:
            return (self._state, 0, True)  # We are done

        if action not in [0, 1]:
            raise ValueError('Invalid action')

        if action == 0:
            movement = np.random.randint(-100, 0)
        else:
            movement = np.random.randint(1, 101)

        self.t_step += 1
        self._state += movement

        if self._state > self._max_right:
            self._state = self._max_right+1
            reward = +1
            self._done = True
        elif self._state < self._max_left:
            self._state = self._max_left-1
            reward = -1
            self._done = True
        else:
            reward = 0
            self._done = False

        obs = self._state
        return (obs, reward, self._done)

    def print_env(self):
        print('Env state: ', self._state)






