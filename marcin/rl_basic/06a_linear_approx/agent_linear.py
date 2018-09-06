import numpy as np
import matplotlib.pyplot as plt
import pdb


class LinearApproximator:
    def __init__(self, step_size):
        self._step_size = step_size
        self._w_st_a0 = 0
        self._b_a0 = 0
        self._w_st_a1 = 0
        self._b_a1 = 0

    
    def estimate(self, state, action):
        if state == 'TERMINAL':
            return 0
        assert isinstance(state, int)
        assert isinstance(action, int) or isinstance(action, np.int64)

        state /= 1000

        if action == 0:
            return self._w_st_a0 * state + self._b_a0
        elif action == 1:
            return self._w_st_a1 * state + self._b_a1
        else:
            raise ValueError('Unknown action')

    def __getitem__(self, key):
        assert isinstance(key, tuple)
        assert len(key) == 2
        return self.estimate(key[0], key[1])
       

    def update(self, state, action, target):
        assert isinstance(state, int)
        assert isinstance(action, int) or isinstance(action, np.int64)

        est = self.estimate(state, action)

        state /= 1000

        if action == 0:
            self._w_st_a0 += self._step_size * (target-est) * state
            self._b_a0 += self._step_size * (target-est)
        elif action == 1:
            self._w_st_a1 += self._step_size * (target-est) * state
            self._b_a1 += self._step_size * (target-est)
        else:
            raise ValueError('Unknown action')


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


class AgentLinear:
    def __init__(self, action_space,
        step_size=0.1, lmbda=None, e_rand=0.0):

        self.Q = LinearApproximator(step_size)

        self._action_space = action_space
        self._step_size = step_size  # usually noted as alpha in literature
        self._discount = 1.0         # usually noted as gamma in literature

        self._lmbda = lmbda          # param for lambda functions

        self._epsilon_random = e_rand  # policy parameter, 0 => always greedy

        self._episode = 0
        self._trajectory = []        # Agent saves history on it's way
        self._eligibility_traces_Q = None   # for lambda funtions
        self._force_random_action = False    # for exploring starts

    def reset(self):
        self._episode += 1
        self._trajectory = []        # Agent saves history on it's way
        #self._eligibility_traces_Q.clear()   # for lambda funtions
        self._force_random_action = False

    def reset_exploring_starts(self):
        self._episode += 1
        self._trajectory = []        # Agent saves history on it's way
        #self._eligibility_traces_Q.clear()   # for lambda funtions
        self._force_random_action = True

    def pick_action(self, obs):
        assert isinstance(obs, int)

        # player_points = obs[1]
        # if player_points < 18:
        #     return 1  # draw
        # else:
        #     return 0  # stick

        if self._force_random_action:
            self._force_random_action = False
            return np.random.choice(self._action_space)
            


        if np.random.rand() < self._epsilon_random:
            # pick random action
            return np.random.choice(self._action_space)

        else:
            # act greedy
            max_Q = float('-inf')
            max_action = None

            possible_actions = []
            for action in self._action_space:
                q = self.Q.estimate(obs, action)
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
        # for act in self._action_space:
        #     if self.Q[last_entry.observation, act] != 0:
        #         raise ValueError('Action from last state has non-zero val.')



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

        """


        Q = self.Q    # Action value array, shape: [world_size, action_space]

        # Shortcuts for more compact notation:

        St = self._trajectory[t].observation      # evaluated state tuple (x, y)
        St_1 = self._trajectory[t+1].observation  # next state tuple (x, y)
        Rt_1 = self._trajectory[t+1].reward       # next step reward
        step = self._step_size
        disc = self._discount

        At = self._trajectory[t].action
        At_1 = self._trajectory[t+1].action
        if At_1 is None:
            At_1 = self.pick_action(St)

        # Q[St, At] = Q[St, At] + step * (Rt_1 + disc * Q[St_1, At_1] - Q[St, At])

        Tt = Rt_1 + disc * self.Q.estimate(St_1, At_1)

        self.Q.update(St, At, Tt)

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

        Q = self.Q    # Action value array, shape: [world_size, action_space]

        EQ = self._eligibility_traces_Q

        St = self._trajectory[t].observation  # current state xy
        St_1 = self._trajectory[t+1].observation
        Rt_1 = self._trajectory[t+1].reward

        
        #
        #   Handle Q
        #

        At = self._trajectory[t].action
        At_1 = self._trajectory[t+1].action

        if At_1 is None:
            At_1 = self.pick_action(St)


        # Update eligibility traces for Q
        EQ.data *= self._lmbda
        EQ[St, At] += 1

        ro_t = Rt_1 + self._discount * Q[St_1, At_1] - Q[St, At]
        Q.data += self._step_size * ro_t * EQ.data

    def eval_td_lambda_offline(self):
        """TD(lambda) update for all states

        Class Params:
            self._lmbda (float, [0, 1]) - param. for weighted average of returns
        """

        if not self._eligibility_traces_Q.is_zeros():
            raise ValueError('TD-lambda offline: eligiblity traces not zeros?')

        self.check_trajectory_terminated_ok()

        # Iterate all states apart from terminal state
        max_t = len(self._trajectory)-2  # inclusive
        for t in range(0, max_t+1):
            self.eval_td_lambda_t(t)

    def eval_td_lambda_online(self):
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

        # Shortcuts for more compact notation:
        St = self._trajectory[t].observation  # current state (x, y)
        Gt = self.calc_Gt(t)            # return for current state

        At = self._trajectory[t].action
        self.Q.update(St, At, Gt)

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


