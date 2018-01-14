import numpy as np
import matplotlib.pyplot as plt
import pdb


class AggregateApproximator:
    def __init__(self, step_size):
        self._step_size = step_size
        
        self._pos_bin_nb = 50
        self._pos_bins = np.linspace(-1.2, 0.5, self._pos_bin_nb+1)

        self._vel_bin_nb = 50
        self._vel_bins = np.linspace(-0.07, 0.07, self._vel_bin_nb+1)

        self._action_nb = 3

        self._states = np.zeros([self._pos_bin_nb,
            self._vel_bin_nb, self._action_nb])

    def _to_idx(self, state, action):
        # print('_to_idx(state, action)', state, type(state), action, type(action))
        assert isinstance(state, tuple)
        assert isinstance(state[0], float)
        assert isinstance(state[1], float)
        assert isinstance(action, int) or isinstance(action, np.int64)

        pos, vel = state[0], state[1]

        # print('pos, vel', pos, vel)

        assert -1.2 <= pos and pos <= 0.5
        assert -0.07 <= vel and vel <= 0.07

        assert action in [-1, 0, 1]
        act_idx = action + 1

        if pos == 0.5:
            return None, None, None

        pos_idx = np.digitize(pos, self._pos_bins) - 1
        if vel == 0.07:
            vel_idx = self._vel_bin_nb-1
        else:
            vel_idx = np.digitize(vel, self._vel_bins) - 1

        # print('pos_idx, vel_idx', pos_idx, vel_idx)

        assert 0 <= pos_idx and pos_idx <= self._pos_bin_nb-1
        assert 0 <= vel_idx and vel_idx <= self._vel_bin_nb-1

        return pos_idx, vel_idx, act_idx

    def estimate(self, state, action):
        pos_idx, vel_idx, act_idx = self._to_idx(state, action)
        if pos_idx is None:
            return 0
        else:
            return self._states[pos_idx, vel_idx, act_idx]
      

    def update(self, state, action, target):
        # print('update(target)', target, type(target))
        assert isinstance(target, float)
        pos_idx, vel_idx, act_idx = self._to_idx(state, action)
        
        est = self.estimate(state, action)
        
        self._states[pos_idx, vel_idx, act_idx] += \
            self._step_size * (target - est)




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
    def __init__(self, action_space, approximator,
        step_size=0.1, e_rand=0.0):

        self.Q = AggregateApproximator(step_size)

        self._action_space = action_space
        self._step_size = step_size  # usually noted as alpha in literature
        self._discount = 1.0         # usually noted as gamma in literature

        self._epsilon_random = e_rand  # policy parameter, 0 => always greedy

        self._episode = 0
        self._trajectory = []        # Agent saves history on it's way

    def reset(self):
        self._episode += 1
        self._trajectory = []        # Agent saves history on it's way


    def pick_action(self, obs):
        assert isinstance(obs, tuple)          


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

