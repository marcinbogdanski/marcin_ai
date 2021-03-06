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
        self.Q = np.zeros([world_size, action_space])

        self.PG = np.random.randn(world_size, action_space) / 100.0

        self._step_size = step_size  # usually noted as alpha in literature
        self._discount = 0.9         # usually noted as gamma in literature
        self._epsilon_random = 0.1
        self._force_random_action = False

        self._lmbda = lmbda          # param for lambda functions

        self._episode = 0
        self._trajectory = []        # Agent saves history on it's way

    def reset(self, expl_start=False):
        self._episode += 1
        self._trajectory = []        # Agent saves history on it's way

        self._force_random_action = expl_start

    def get_pg_prob(self):
        return self.softmax(self.PG)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def pick_action(self, obs):
        assert isinstance(obs, int)
        # Randomly go left or right (0 is left, 1 is right)
        #return np.random.choice([0, 1])

        if self._force_random_action:
            self._force_random_action = False
            return np.random.choice([0, 1])

        # policy = 'e-greedy'
        policy = 'policy_gradient'

        if policy == 'e-greedy':
            if np.random.rand() < self._epsilon_random:
                return np.random.choice([0, 1])
            else:
                Ql = self.Q[obs][0]
                Qr = self.Q[obs][1]

                if Ql == Qr:
                    return np.random.choice([0, 1])
                elif Ql > Qr:
                    return 0  # go left
                else:
                    return 1  # go rigth
        elif policy == 'policy_gradient':
            
            obs_one_hot = np.zeros(len(self.PG))
            obs_one_hot[obs] = 1

            act_h = np.dot(obs_one_hot, self.PG)
            prob = self.softmax(act_h)

            act = np.random.choice(range(2), p=prob)

            return act
            

        else:
            raise ValueError('This should not happen')


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

        """

        assert not self._trajectory[t].done

        V = self.V    # State values array, shape: [world_size]
        Q = self.Q    # Action value array, shape: [world_size, action_space]

        # Shortcuts for more compact notation:

        St = self._trajectory[t].observation      # evaluated state tuple (x, y)
        St_1 = self._trajectory[t+1].observation  # next state tuple (x, y)
        Rt_1 = self._trajectory[t+1].reward       # next step reward
        done = self._trajectory[t+1].done
        step = self._step_size
        disc = self._discount

        if not done:
            V[St] = V[St] + step * (Rt_1 + disc*V[St_1] - V[St])
        else:
            V[St] = V[St] + step * (Rt_1 - V[St])

        At = self._trajectory[t].action
        At_1 = self._trajectory[t+1].action
        
        if At_1 is None:
            At_1 = self.pick_action(St)

        if not done:
            Q[St, At] = Q[St, At] + step*(Rt_1 + disc*Q[St_1, At_1] - Q[St, At])
        else:
            Q[St, At] = Q[St, At] + step * (Rt_1 - Q[St, At])



        # policy gradient calc   # old

        # state/action feature vector
        X = np.zeros(self.PG.shape)
        X[St, At] = 1

        h_all_act = self.PG[St]
        assert h_all_act.ndim == 1
        assert len(h_all_act) == 2

        prob_all_act = self.softmax(h_all_act)
        assert prob_all_act.ndim == 1
        assert len(prob_all_act) == 2

        ln_grad = np.copy(X)
        ln_grad[St] -= prob_all_act

        # Monte Carlo
        # update = self._step_size * Gt * ln_grad

        # Actor-Critic
        #update = self._step_size * Q[St, At] * ln_grad

        # AC with advantage function (baseline)
        # Adv = Q[St, At] - V[St]
        # update = self._step_size * Adv * ln_grad

        # AC with TD error as target
        ro = Rt_1 + self._discount * V[St_1] - V[St]
        update = self._step_size * ro * ln_grad

        self.PG += update




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

        assert not self._trajectory[t].done

        V = self.V    # State values array, shape: [world_size]
        Q = self.Q    # Action value array, shape: [world_size, action_space]

        # Shortcuts for more compact notation:
        St = self._trajectory[t].observation  # current state (x, y)
        Gt = self.calc_Gt(t)            # return for current state

        V[St] = V[St] + self._step_size * (Gt - V[St])

        At = self._trajectory[t].action
        Q[St, At] = Q[St, At] + self._step_size * (Gt - Q[St, At])

        # policy gradient calc

        # state/action feature vector
        X = np.zeros(self.PG.shape)
        X[St, At] = 1

        h_all_act = self.PG[St]
        assert h_all_act.ndim == 1
        assert len(h_all_act) == 2

        prob_all_act = self.softmax(h_all_act)
        assert prob_all_act.ndim == 1
        assert len(prob_all_act) == 2

        ln_grad = np.copy(X)
        ln_grad[St] -= prob_all_act


        # Monte Carlo
        update = self._step_size * Gt * ln_grad

        self.PG += update


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

