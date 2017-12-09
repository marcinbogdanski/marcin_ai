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

    real_values_19 = np.arange(-20, 22, 2) / 20.0
    real_values_19[0] = real_values_19[-1] = 0

    GROUND_TRUTH = {
        5: [0, -2/3, -1/3, 0, 1/3, 2/3, 0],  # Actual state-values
        19: real_values_19
    }

    def __init__(self, size):
        self._max_left = 1          # last valid non-terminal state to the left
        self._max_right = size      # last non-terminal state to the right
        self._start_state = (size // 2) + 1
        self.reset()

    def reset(self):
        self.t_step = 0
        self._state = self._start_state
        self._done = False

        return self._state

    def step(self, action):
        if self._done:
            return (self._state, 0, True)  # We are done

        if action not in [-1, 1]:
            raise ValueError('Invalid action')

        self.t_step += 1
        self._state += action

        obs = self._state
        if self._state > self._max_right:
            reward = +1
            self._done = True
        elif self._state < self._max_left:
            reward = -1
            self._done = True
        else:
            reward = 0
            self._done = False

        return (obs, reward, self._done)

    def print_env(self):
        print('Env state: ', self._state)






