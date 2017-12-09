import numpy as np
import matplotlib.pyplot as plt
import pdb

class HistoryData:
    """One piece of agent trajectory"""
    def __init__(self, t_step, observation, reward, done):
        self.t_step = t_step
        self.observation = observation
        self.reward = reward
        self.action = None
        self.done = done

    def __str__(self):
        return '{0}: {1}, {2} {3}   {4}'.format(
            self.t_step, self.observation, self.reward, self.done, self.action)


class Agent:
    def __init__(self, size, step_size=0.1):
        self.V = np.zeros([size])
        self.V[0] = 0   # initialise state-values of terminal states to zero!
        self.V[-1] = 0

        self.step = step_size   # step-size parameter - usually noted as alpha
        self.disc = 1.0    # discount factor - usually noted as gamma

        self.reset()

    def reset(self):
        self.trajectory = []   # Agent saves history on it's way

    def pick_action(self, obs):
        # Randomly go left or right
        return np.random.choice([-1, 1])

    def append_trajectory(self, t_step, prev_action, observation, reward, done):
        if len(self.trajectory) != 0:
            self.trajectory[-1].action = prev_action

        self.trajectory.append(
            HistoryData(t_step, observation, reward, done))

    def print_trajectory(self):
        print('Trajectory:')
        for element in self.trajectory:
            print(element)
        print('Total trajectory steps: {0}'.format(len(self.trajectory)))

    def eval_td_t(self, t):
        """TD update state-value for single state in trajectory

        This assumesss time step t+1 is availalbe in the trajectory

        For online updates:
            Call with t equal to previous time step
        For offline updates:
            Iterate trajectory from t=0 to t=T-1 and call for every t
        """

        # Shortcuts for more compact notation:

        V = self.V              # State values array, shape: [size_x, size_y]
        St = self.trajectory[t].observation      # evaluated state tuple (x, y)
        St_1 = self.trajectory[t+1].observation  # next state tuple (x, y)
        Rt_1 = self.trajectory[t+1].reward       # next step reward

        V[St] = V[St] + self.step*(Rt_1 + self.disc*V[St_1] - V[St])

        # print('self.V', np.round(self.V,4))

    def eval_td_online(self):
        self.eval_td_t(len(self.trajectory)-2)  # Eval next-to last state

    def eval_td_offline(self):
        # Do MC update only if episode terminated
        if self.trajectory[-1].done:

            # Iterate all states in trajectory
            for t in range(0, len(self.trajectory)-1):
                # Update state-value at time t
                self.eval_td_t(t)


    def eval_nstep_t(self, t, n):
        """n-step update state-value for single state in trajectory

        This assumesss time steps t+1 to t+n are availalbe in the trajectory

        For online updates:
            Call with t equal to n-1 time step,
            Then at termination, call for each of remaining steps including T-1
        For offline updates:
            Iterate trajectory from t=0 to t=T-1 and call for every t
        """
        # Shortcuts for more compact notation:
        V = self.V              # State values array, shape: [size_x, size_y]
        St = self.trajectory[t].observation      # evaluated state tuple (x, y)
        Gt = 0            # return for current state
        
        # print('t = ', t)

        T = len(self.trajectory)-1
        max_j = min(t+n, T)
        # First cell to iterate is t+1, last cell is t+n inclusive (or T)
        discount = 1.0

        # print('T = ', T)
        # print('max_j = ', max_j)
        # print('loop: ')

        for j in range(t+1, max_j+1):
            # print('  j = ', j)
            Rj = self.trajectory[j].reward
            # print('  Rj = ', Rj)
            # print('  discount = ', discount)
            Gt += discount * Rj
            # print('  Gt = ', Gt)
            discount *= self.disc
            # print('  in loop discount', discount)


        # Note that V[Sj] will have state-value of state n+t or
        # zero if n+t >= T
        Sj = self.trajectory[j].observation
        # print('Sj = ', Sj)
        # print('discount = ', discount)
        Gt += discount * V[Sj]

        V[St] = V[St] + self.step*(Gt - V[St])

        # print('copy(V)', np.round(V,4))


    def eval_nstep_online(self, n):
        start_t = len(self.trajectory)-n-1
        if start_t >= 0:
            self.eval_nstep_t(start_t, n)
        self.eval_td_online()

    def eval_nstep_offline(self, n):
        if self.trajectory[-1].done:
            for t in range(0, len(self.trajectory)-1):
                self.eval_nstep_t(t, n)
        

    def eval_mc_t(self, t):
        """MC update state-values for single state in trajectory

        This assumes episode is completed and trajectory is present
        from start to termination.

        For online updates:
            N/A

        For offline updates:
            Iterate trajectory from t=0 to t=T-1 and call for every t

        """
        # Shortcuts for more compact notation:
        V = self.V             # State values array, shape: [size_x, size_y]
        St = self.trajectory[t].observation  # current state (x, y)
        Gt = 0            # return for current state

        # Iterate all states after this one
        for j in range(t+1, len(self.trajectory)):
            Rj = self.trajectory[j].reward   # reward at time j
            Gt += self.disc**j * Rj          # add with discount

        V[St] = V[St] + self.step*(Gt - V[St])

    def eval_mc_offline(self):
        # Do MC update only if episode terminated
        if self.trajectory[-1].done:

            # Iterate all states in trajectory
            for t in range(0, len(self.trajectory)-1):
                # Update state-value at time t
                self.eval_mc_t(t)