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


class AgentPG:
    def __init__(self, world_size, action_space,
        step_size=0.1, lmbda=None):
        
        self.V = np.zeros([world_size])
        self.V[0] = 0   # initialise state-values of terminal states to zero!
        self.V[-1] = 0

        self.Q = np.zeros([world_size, action_space])
        self.Q[0, 0] = 0   # initialise action-values of actions from
        self.Q[0, 1] = 0   # terminal state to zero
        self.Q[-1, 0] = 0
        self.Q[-1, 1] = 0

        self._step_size = step_size  # usually noted as alpha in literature
        self._discount = 1.0         # usually noted as gamma in literature

        self._lmbda = lmbda          # param for lambda functions

        self._episode = 0
        self._trajectory = []        # Agent saves history on it's way
        self._eligibility_traces_V = np.zeros_like(self.V)  # for lambda funtions
        self._eligibility_traces_Q = np.zeros_like(self.Q)   # for lambda funtions

    def reset(self):
        self._episode += 1
        self._trajectory = []        # Agent saves history on it's way
        self._eligibility_traces_V.fill(0)
        self._eligibility_traces_Q.fill(0)   # for lambda funtions

    def pick_action(self, obs):
        # Randomly go left or right (0 is left, 1 is right)
        return np.random.choice([0, 1])

    def append_trajectory(self, t_step, prev_action, observation, reward, done):
        if len(self._trajectory) != 0:
            self._trajectory[-1].action = prev_action

        self._trajectory.append(
            HistoryData(t_step, observation, reward, done))

    def print_trajectory(self):
        print('Trajectory:')
        for element in self._trajectory:
            print(element)
        print('Total trajectory steps: {0}'.format(len(self._trajectory)))









    def eval_td_t(self, t):
        """TD update state-value for single state in trajectory

        This assumesss time step t+1 is availalbe in the trajectory

        For online updates:
            Call with t equal to previous time step

        For offline updates:
            Iterate trajectory from t=0 to t=T-1 and call for every t

        Params:
            t (int [t, T-1]) - time step in trajectory,
                    0 is initial state; T-1 is last non-terminal state

            V (float arr) - optional,
                    if passed, funciton will operate on this array
                    if None, then function will operate on self.V
        """


        V = self.V    # State values array, shape: [world_size]
        Q = self.Q    # Action value array, shape: [world_size, action_space]

        # Shortcuts for more compact notation:

        St = self._trajectory[t].observation      # evaluated state tuple (x, y)
        St_1 = self._trajectory[t+1].observation  # next state tuple (x, y)
        Rt_1 = self._trajectory[t+1].reward       # next step reward
        step = self._step_size
        disc = self._discount

        V[St] = V[St] + step * (Rt_1 + disc*V[St_1] - V[St])

        At = self._trajectory[t].action
        At_1 = self._trajectory[t+1].action
        if At_1 is None:
            At_1 = self.pick_action(St)

        Q[St, At] = Q[St, At] + step * (Rt_1 + disc * Q[St_1, At_1] - Q[St, At])

    def eval_td_online(self):
        self.eval_td_t(len(self._trajectory) - 2)  # Eval next-to last state

    def eval_td_offline(self):
        """ Do TD update for all states in trajectory

        Note:
            This updates V and Q arrays "in place". True offline update should
            update copy of V (or Q), then replace V with a copy at the end.
            This function will yeild slightly different result.

        """
        
        if not self._trajectory[-1].done:
            raise ValueError('Cant do offline on non-terminated episode')

        # Iterate all states in trajectory
        for t in range(0, len(self._trajectory)-1):
            # Update state-value at time t
            self.eval_td_t(t)







    def eval_td_lambda_t(self, t):
        """TD(lambda) update for particular state.0

        Note:
            Becouse this function builds eligibility trace dictionary in order,
            it MUST be called in correct sequence, from t=0 to T=T-1.
            It can be called only once per t-step

        For online updates:
            Call with t equal to previous time step

        For offline updates:
            Iterate trajectory from t=0 to t=T-1 and call for every t

        Params:
            t (int [t, T-1]) - time step in trajectory,
                    0 is initial state; T-1 is last non-terminal state

        """

        V = self.V    # State values array, shape: [world_size]
        Q = self.Q    # Action value array, shape: [world_size, action_space]

        EV = self._eligibility_traces_V   # eligibility trace dictionary
        EQ = self._eligibility_traces_Q

        St = self._trajectory[t].observation  # current state xy
        St_1 = self._trajectory[t+1].observation
        Rt_1 = self._trajectory[t+1].reward

        #
        #   Handle V
        #

        # Update eligibility traces for V
        EV *= self._lmbda
        EV[St] += 1

        ro_t = Rt_1 + self._discount * V[St_1] - V[St]
        V += self._step_size * ro_t * EV

        #
        #   Handle Q
        #

        At = self._trajectory[t].action
        At_1 = self._trajectory[t+1].action

        if At_1 is None:
            At_1 = self.pick_action(St)


        # Update eligibility traces for Q
        EQ *= self._lmbda
        EQ[St, At] += 1

        ro_t = Rt_1 + self._discount * Q[St_1, At_1] - Q[St, At]
        Q += self._step_size * ro_t * EQ

    def eval_td_lambda_offline(self):
        """TD(lambda) update for all states

        Class Params:
            self._lmbda (float, [0, 1]) - param. for weighted average of returns
        """

        if np.count_nonzero(self._eligibility_traces_V) != 0:
            raise ValueError('TD-lambda offline: eligiblity traces not empty?')
        if np.count_nonzero(self._eligibility_traces_Q) != 0:
            raise ValueError('TD-lambda offline: eligiblity traces not empty?')

        # Do offline update only if episode terminated
        if not self._trajectory[-1].done:
            raise ValueError('Cant do offline on non-terminated episode')

        # Iterate all states apart from terminal state
        max_t = len(self._trajectory)-2  # inclusive
        for t in range(0, max_t+1):
            self.eval_td_lambda_t(t)

    def eval_td_lambda_online(self, V=None):
        t = len(self._trajectory) - 2   # Previous time step
        self.eval_td_lambda_t(t)










    def calc_Gt(self, t):
        """Calculates return for state t

        Params:
            t (int [t, T-1]) - time step in trajectory,
                    0 is initial state; T-1 is last non-terminal state

        """

        T = len(self._trajectory)-1   # terminal state
        discount = 1.0

        Gt = 0

        # Iterate from t+1 to T (inclusive on both start and finish)
        for j in range(t+1, T+1):
            Rj = self._trajectory[j].reward
            Gt += discount * Rj
            discount *= self._discount

        return Gt

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
            t (int [t, T-1]) - time step in trajectory,
                    0 is initial state; T-1 is last non-terminal state

            V (float arr) - optional,
                    if passed, funciton will operate on this array
                    if None, then function will operate on self.V

        """

        assert not self._trajectory[t].done

        V = self.V    # State values array, shape: [world_size]
        Q = self.Q    # Action value array, shape: [world_size, action_space]

        # Shortcuts for more compact notation:
        St = self._trajectory[t].observation  # current state (x, y)
        Gt = self.calc_Gt(t)            # return for current state

        V[St] = V[St] + self._step_size * (Gt - V[St])

        At = self._trajectory[t].action
        Q[St, At] = Q[St, At] + self._step_size * (Gt - Q[St, At])

    def eval_mc_offline(self):
        """MC update for all statates. Call after episode terminates

        Note:
            This updates V array "in place". True offline update should
            update copy of V, then replace V with a copy at the end.
            This function will yeild slightly different result.
        """

        # Do MC update only if episode terminated
        if not self._trajectory[-1].done:
            raise ValueError('Cant do offline on non-terminated episode')

        # Iterate all states in trajectory, apart from terminal state
        T = len(self._trajectory) - 1
        for t in range(0, T):
            # Update state-value at time t
            self.eval_mc_t(t)

