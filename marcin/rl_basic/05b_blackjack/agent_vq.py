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


class AgentVQ:
    def __init__(self, state_space, action_space,
        step_size=0.1, nb_steps=None, lmbda=None):

        self.V = {}
        self.Q = {}

        self.Q_num = {}  # number of times state-action visited

        for state in state_space:
            self.V[state] = 0
            for action in action_space:
                self.Q[state, action] = 0
                self.Q_num[state, action] = 0
        
        self._action_space = action_space
        self._step_size = step_size  # usually noted as alpha in literature
        self._discount = 1.0         # usually noted as gamma in literature

        self._lmbda = lmbda          # param for lambda functions

        self._episode = 0
        self._trajectory = []        # Agent saves history on it's way
        self._eligibility_traces_V = {}   # for lambda funtions
        self._eligibility_traces_Q = {}   # for lambda funtions
        self._force_random_action = False  # for exploring starts

    def reset(self):
        self._episode += 1
        self._trajectory = []        # Agent saves history on it's way
        self._eligibility_traces_V = {}
        self._eligibility_traces_Q = {}   # for lambda funtions
        self._force_random_action = False

    def reset_exploring_starts(self):
        self._episode += 1
        self._trajectory = []        # Agent saves history on it's way
        self._eligibility_traces_V = {}
        self._eligibility_traces_Q = {}   # for lambda funtions
        self._force_random_action = True

    def pick_action(self, obs):

        # player_points = obs[1]
        # if player_points < 18:
        #     return 1  # draw
        # else:
        #     return 0  # stick

        if self._force_random_action:
            self._force_random_action = False
            return np.random.choice(self._action_space)
            


        if np.random.rand() < 0.00:
            # pick random action
            return np.random.choice(self._action_space)

        else:
            # act greedy
            max_Q = float('-inf')
            max_action = None

            possible_actions = []
            for action in self._action_space:
                q = self.Q[obs, action]
                if q > max_Q:
                    possible_actions.clear()
                    possible_actions.append(action)
                    max_Q = q
                elif q == max_Q:
                    possible_actions.append(action)
            return np.random.choice(possible_actions)

            

    def append_trajectory(self, t_step, prev_action, observation, reward, done):
        if len(self._trajectory) != 0:
            self._trajectory[-1].action = prev_action
            self.Q_num[self._trajectory[-1].observation, prev_action] += 1

        self._trajectory.append(
            HistoryData(t_step, observation, reward, done))

    def print_trajectory(self):
        print('Trajectory:')
        for element in self._trajectory:
            print(element)
        print('Total trajectory steps: {0}'.format(len(self._trajectory)))

    def check_trajectory_terminated_ok(self):
        last_entry = self._trajectory[-1]
        if not last_entry.done:
            raise ValueError('Cant do offline on non-terminated episode')
        if self.V[last_entry.observation] != 0:
            raise ValueError('Last state in trajectory has non-zero value')
            for act in self.action_space:
                if self.Q[last_entry.observation, act] != 0:
                    raise ValueError('Action from last state has non-zero val.')



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
        self.check_trajectory_terminated_ok()

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

        if St not in EV:
            EV[St] = 0

        # Update eligibility traces for V
        for s in EV:
            EV[s] *= self._lmbda
        EV[St] += 1

        ro_t = Rt_1 + self._discount * V[St_1] - V[St]
        for s in EV:
            V[s] = V[s] + self._step_size * ro_t * EV[s]

        #
        #   Handle Q
        #

        At = self._trajectory[t].action
        At_1 = self._trajectory[t+1].action

        if At_1 is None:
            At_1 = self.pick_action(St)

        if (St, At) not in EQ:
            EQ[St, At] = 0

        # Update eligibility traces for Q
        for s, a in EQ:
            EQ[s, a] *= self._lmbda
        EQ[St, At] += 1

        ro_t = Rt_1 + self._discount * Q[St_1, At_1] - Q[St, At]
        for s, a in EQ:
            Q[s, a] = Q[s, a] + self._step_size * ro_t * EQ[s, a]

    def eval_td_lambda_offline(self):
        """TD(lambda) update for all states

        Class Params:
            self._lmbda (float, [0, 1]) - param. for weighted average of returns
        """

        if len(self._eligibility_traces_V) != 0:
            raise ValueError('TD-lambda offline: eligiblity traces not empty?')
        if len(self._eligibility_traces_Q) != 0:
            raise ValueError('TD-lambda offline: eligiblity traces not empty?')

        self.check_trajectory_terminated_ok()

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

    def eval_mc_t(self, t):
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

        """

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

        self.check_trajectory_terminated_ok()

        # Iterate all states in trajectory, apart from terminal state
        T = len(self._trajectory) - 1
        for t in range(0, T):
            # Update state-value at time t
            self.eval_mc_t(t)

