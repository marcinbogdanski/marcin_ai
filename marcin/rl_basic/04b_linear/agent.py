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

        self._episode = 0
        self.trajectory = []   # Agent saves history on it's way

    def reset(self):
        self._episode += 1
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

    def calc_Gt(self, t, n=float('inf')):
        """Calculates return for state t, using n future steps.
        
        If n >= T (terminal state), then calculates full return for state t

        For n == 1 this equals to TD return
        For n == +inf this equals to MC return
        """

        T = len(self.trajectory)-1   # terminal state
        max_j = min(t+n, T)    # last state iterated, inclusive
        discount = 1.0

        Gt = 0

        # Iterate from t+1 to t+n or T (inclusive on both start and finish)
        #print('calc_GT()  t, n = ', t, n)
        for j in range(t+1, max_j+1):
            Rj = self.trajectory[j].reward
            Gt += discount * Rj
            discount *= self.disc

        # Note that V[Sj] will have state-value of state n+t or
        # zero if n+t >= T as V[St=T] must equal 0
        Sj = self.trajectory[j].observation
        Gt += discount * self.V[Sj]

        return Gt

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
        Gt = self.calc_Gt(t, n=n)

        V[St] = V[St] + self.step*(Gt - V[St])

    # def eval_nstep_online(self, n):
    #     start_t = len(self.trajectory)-n-1
    #     if start_t >= 0:
    #         self.eval_nstep_t(start_t, n)
    #     self.eval_td_online()

    def eval_nstep_offline(self, n):
        if self.trajectory[-1].done:
            for t in range(0, len(self.trajectory)-1):
                self.eval_nstep_t(t, n)
        

    def eval_lambda_return_t(self, t, lmbda, V=None):
        """TD-Lambda update state-value for single state in trajectory

        This assumesss time steps from t+1 to terminating state
        are available in the trajectory

        For online updates:
            N/A
        For offline updates:
            Iterate trajectory from t=0 to t=T-1 and call for every t
        """

        if V is None:
            V = self.V             # State values array, shape: [size_x, size_y]

        lambda_trunctuate = 1e-3   # discard if lmbda drops below this
        
        # Shortcuts for more compact notation:
        
        St = self.trajectory[t].observation      # evaluated state tuple (x, y)
        Gt_lambda = 0           # weigthed averated return for current state
        
        T = len(self.trajectory)-1
        max_n = T-t-1           # inclusive

        lmbda_iter = 1
        for n in range(1, max_n+1):

            Gtn = self.calc_Gt(t, n=n)
            Gt_lambda += lmbda_iter * Gtn
            lmbda_iter *= lmbda

            if lmbda_iter < lambda_trunctuate:
                break

        Gt_lambda *= (1 - lmbda)

        if lmbda_iter >= lambda_trunctuate:
            Gt_lambda += lmbda_iter * self.calc_Gt(t)

        V[St] = V[St] + self.step*(Gt_lambda - V[St])

        return V

    def eval_lambda_return_offline(self, lmbda, V=None):
        if V is None:
            V = self.V

        # Do TD-lambda update only if episode terminated
        if self.trajectory[-1].done:

            # Iterate all states in trajectory
            for t in range(0, len(self.trajectory)-1):
                # Update state-value at time t
                self.eval_lambda_return_t(t, lmbda, V)

        return V

    def eval_td_lambda_offline(self, lmbda, V=None):
        """TD(lambda) update for all states"""
        
        if V is None:
            V = self.V

        E = {}  # eligibility trace dictionary

        # Do TD-lambda update only if episode terminated
        if not self.trajectory[-1].done:
            return

        # Iterate all states apart from terminal state
        max_t = len(self.trajectory)-2  # inclusive
        for t in range(0, max_t+1):
            St = self.trajectory[t].observation
            E[St] = 0

        # Iterate all states apart from terminal state
        for t in range(0, max_t+1):
            St = self.trajectory[t].observation  # current state xy
            St_1 = self.trajectory[t+1].observation
            Rt_1 = self.trajectory[t+1].reward

            # Update eligibility traces
            for s in E:
                E[s] *= lmbda
            E[St] += 1

            ro_t = Rt_1 + self.disc * V[St_1] - V[St]
            for s in E:
                V[s] = V[s] + self.step * ro_t * E[s]

        return V


    def eval_mc_t(self, t, V=None):
        """MC update for state-values for single state in trajectory

        Note:
            This assumes episode is completed and trajectory is present
            from start to termination.

        For online updates:
            N/A

        For offline updates:
            Iterate trajectory from t=0 to t=T-1 and call for every t

        Params:
            t (int) - time step in trajectory,
                      0 is initial state; T-1 is last non-terminal state
            V (float arr) - optional,
                    if passed, funciton will operate on this array
                    if None, then function will operate on self.V

        """

        if V is None:
            V = self.V   # State values array, shape: [size_x, size_y]

        # Shortcuts for more compact notation:
        St = self.trajectory[t].observation  # current state (x, y)
        Gt = self.calc_Gt(t)                 # return for current state

        V[St] = V[St] + self.step*(Gt - V[St])

    def eval_mc_offline(self, V=None):
        """MC update for all statates. Call after episode terminates

        Note:
            This updates V array "in place". True offline update should
            update copy of V, then replace V with a copy at the end.
            This function will yeild slightly different result.

        Params:
            V (float arr) - optional,
                    if passed, funciton will operate on this array
                    if None, then function will operate on self.V
        """

        if V is None:
            V = self.V

        # Do MC update only if episode terminated
        if not self.trajectory[-1].done:
            raise ValueError('Cant do MC update on non-terminated episode')

        # Iterate all states in trajectory, including terminal state
        for t in range(0, len(self.trajectory)-1):
            # Update state-value at time t
            self.eval_mc_t(t, V)

