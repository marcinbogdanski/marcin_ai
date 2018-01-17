import numpy as np
import matplotlib.pyplot as plt
import collections
import pdb

import tile_coding
import neural_mini



class AggregateApproximator:
    def __init__(self, step_size):
        self._step_size = step_size
        
        self._pos_bin_nb = 64
        self._pos_bins = np.linspace(-1.2, 0.5, self._pos_bin_nb+1)

        self._vel_bin_nb = 64
        self._vel_bins = np.linspace(-0.07, 0.07, self._vel_bin_nb+1)

        self._action_nb = 3

        self._states = np.zeros([self._pos_bin_nb,
            self._vel_bin_nb, self._action_nb])

        max_len = 2000
        self._hist_pos = collections.deque(maxlen=max_len)
        self._hist_vel = collections.deque(maxlen=max_len)
        self._hist_act = collections.deque(maxlen=max_len)
        self._hist_tar = collections.deque(maxlen=max_len)

    def reset(self):
        self._hist_pos.clear()
        self._hist_vel.clear()
        self._hist_act.clear()
        self._hist_tar.clear()

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

        self._hist_pos.append(state[0])
        self._hist_vel.append(state[1])
        self._hist_act.append(action)
        self._hist_tar.append(target)
        
        est = self.estimate(state, action)
        
        self._states[pos_idx, vel_idx, act_idx] += \
            self._step_size * (target - est)


class TileApproximator:

    def __init__(self, step_size):
        self._num_of_tillings = 8
        self._step_size = step_size / self._num_of_tillings

        self._pos_scale = self._num_of_tillings / (0.5 + 1.2)
        self._vel_scale = self._num_of_tillings / (0.07 + 0.07)

        self._hashtable = tile_coding.IHT(2048)

        self._weights = np.zeros(2048)
        # self._weights = np.random.randn(2048) - 100


        max_len = 2000
        self._hist_pos = collections.deque(maxlen=max_len)
        self._hist_vel = collections.deque(maxlen=max_len)
        self._hist_act = collections.deque(maxlen=max_len)
        self._hist_tar = collections.deque(maxlen=max_len)

    def reset(self):
        pass
        # self._hist_pos.clear()
        # self._hist_vel.clear()
        # self._hist_act.clear()
        # self._hist_tar.clear()

    def _test_input(self, state, action):
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

        if pos == 0.5:
            return None, None, None

        return pos, vel, action

    def estimate(self, state, action):
        pos, vel, action = self._test_input(state, action)
        if pos is None:
            return 0  # terminal state

        active_tiles = tile_coding.tiles(
            self._hashtable, self._num_of_tillings,
            [self._pos_scale * pos, self._vel_scale * vel],
            [action])

        return np.sum(self._weights[active_tiles])


    def update(self, state, action, target):
        pos, vel, action = self._test_input(state, action)
        if pos is None:
            return 0  # terminal state

        self._hist_pos.append(pos)
        self._hist_vel.append(vel)
        self._hist_act.append(action)
        self._hist_tar.append(target)

        active_tiles = tile_coding.tiles(
            self._hashtable, self._num_of_tillings,
            [self._pos_scale * pos, self._vel_scale * vel],
            [action])

        est = np.sum(self._weights[active_tiles])

        delta = self._step_size * (target - est)

        for tile in active_tiles:
            self._weights[tile] += delta



class NeuralApproximator:

    def __init__(self, step_size, discount):
        self._step_size = step_size
        self._discount = discount

        self._nn = neural_mini.NeuralNetwork2([2, 128, 3])

        self._pos_offset = 0.35
        self._pos_scale = 2 / 1.7  # -1.2 to 0.5 should be for NN
        self._vel_scale = 2 / 0.14  # maps vel to -1..1

        max_len = 500
        self._hist_pos = collections.deque(maxlen=max_len)
        self._hist_vel = collections.deque(maxlen=max_len)
        self._hist_act = collections.deque(maxlen=max_len)
        self._hist_tar = collections.deque(maxlen=max_len)

        self._hist_rew_next = collections.deque(maxlen=max_len)
        self._hist_pos_next = collections.deque(maxlen=max_len)
        self._hist_vel_next = collections.deque(maxlen=max_len)

        self._q_back = collections.deque(maxlen=50)
        self._q_stay = collections.deque(maxlen=50)
        self._q_fwd = collections.deque(maxlen=50)

    def reset(self):
        pass
        # self._hist_pos.clear()
        # self._hist_vel.clear()
        # self._hist_act.clear()
        # self._hist_tar.clear()


    def _test_input(self, state, action):
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

        if pos == 0.5:
            return None, None, None

        return pos, vel, action

    def estimate(self, state, action):
        pos, vel, action = self._test_input(state, action)
        if pos is None:
            return 0  # terminal state

        pos += self._pos_offset
        pos *= self._pos_scale
        vel *= self._vel_scale

        est = self._nn.forward(np.array([[pos, vel]]))

        assert action in [-1, 0, 1]

        return est[0, action+1]

    def update(self, state, action, target):
        self.update_replay(state, action, target)

    def update_single(self, state, action, target):
        pos, vel, action = self._test_input(state, action)
        if pos is None:
            return 0  # terminal state

        # pdb.set_trace()

        self._hist_pos.append(pos)
        self._hist_vel.append(vel)
        self._hist_act.append(action)
        self._hist_tar.append(target)

        pos += self._pos_offset
        pos *= self._pos_scale
        vel *= self._vel_scale

        est = self._nn.forward(np.array([[pos, vel]]))

        # do not update other action predicitons
        assert action in [-1, 0, 1]
        est[0, action+1] = target



        batch = [ (np.array([[pos, vel]]), est) ]

        self._nn.train_batch(batch, self._step_size)

        
    def update_replay(self, state, action, target):
        pos, vel, action = self._test_input(state, action)
        if pos is None:
            return 0  # terminal state

        # pdb.set_trace()

        self._hist_pos.append(pos)
        self._hist_vel.append(vel)
        self._hist_act.append(action)
        self._hist_tar.append(target)

        if len(self._hist_pos) < 32:
            return

        idx = np.random.choice(range(len(self._hist_pos)), 32)

        batch = []
        for i in idx:
            pp = self._hist_pos[i]
            vv = self._hist_vel[i]
            aa = self._hist_act[i]
            tt = self._hist_tar[i]

            pp += self._pos_offset
            pp *= self._pos_scale
            vv *= self._vel_scale

            est = self._nn.forward(np.array([[pp, vv]]))
            assert aa in [-1, 0, 1]
            est[0, aa+1] = tt

            batch.append( (np.array([[pp, vv]]), np.array(est)) )

        self._nn.train_batch(batch, self._step_size)

    def update2(self, state, action, reward, state_next):
        pos, vel, action = self._test_input(state, action)
        if pos is None:
            return 0  # current state is terminal

        # if state_next[0] == 0.5:
        #     pdb.set_trace()

        self._hist_pos.append(pos)
        self._hist_vel.append(vel)
        self._hist_act.append(action)
        self._hist_tar.append(0)

        assert reward in [-1, 0]
        pos_next = state_next[0]
        vel_next = state_next[1]

        assert -1.2 <= pos_next and pos_next <= 0.5
        assert -0.07 <= vel_next and vel_next <= 0.07

        if pos_next == 0.5:
            assert reward == 0

        self._hist_rew_next.append(reward)
        self._hist_pos_next.append(pos_next)
        self._hist_vel_next.append(vel_next)

        if len(self._hist_pos) < 32:
            return

        idx = np.random.choice(range(len(self._hist_pos)), 32)
        idx[31] = len(self._hist_pos) - 1

        batch = []
        for i in idx:
            pp = self._hist_pos[i]
            vv = self._hist_vel[i]
            aa = self._hist_act[i]

            rr = self._hist_rew_next[i]
            pp_n = self._hist_pos_next[i]
            vv_n = self._hist_vel_next[i]

            pp += self._pos_offset
            pp *= self._pos_scale
            vv *= self._vel_scale

            pp_n += self._pos_offset
            pp_n *= self._pos_scale
            vv_n *= self._vel_scale

            est = self._nn.forward(np.array([[pp, vv]]))
            est_n = self._nn.forward(np.array([[pp_n, vv_n]]))
            q_n = np.max(est_n)

            if pp_n == 0.5:
                pdb.set_trace()
                # next state is terminal
                tt = rr 
            else:
                tt = rr + self._discount * q_n

            assert aa in [-1, 0, 1]
            est[0, aa+1] = tt

            batch.append( (np.array([[pp, vv]]), np.array(est)) )

        self._nn.train_batch(batch, self._step_size)



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

        if approximator == 'aggregate':
            self.Q = AggregateApproximator(step_size)
        elif approximator == 'tile':
            self.Q = TileApproximator(step_size)
        elif approximator == 'neural':
            self.Q = NeuralApproximator(step_size, 0.99)
        else:
            raise ValueError('Unknown approximator')

        self._action_space = action_space
        self._step_size = step_size  # usually noted as alpha in literature
        self._discount = 0.99         # usually noted as gamma in literature

        self._epsilon_random = e_rand  # policy parameter, 0 => always greedy

        self._episode = 0
        self._trajectory = []        # Agent saves history on it's way

    def reset(self):
        self._episode += 1
        self._trajectory = []        # Agent saves history on it's way

        self.Q.reset()


    def pick_action(self, obs):
        assert isinstance(obs, tuple)          


        if np.random.rand() < self._epsilon_random:
            # pick random action
            res = np.random.choice(self._action_space)

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
            res = np.random.choice(possible_actions)

        return res



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

        if isinstance(self.Q, NeuralApproximator):
            self.Q.update2(St, At, Rt_1, St_1)

            # if St_1[0] == 0.5:
            #     pdb.set_trace()
        else:
            self.Q.update(St, At, Tt)

    def eval_td_online(self):
        self.eval_td_t(len(self._trajectory) - 2)  # Eval next-to last state

