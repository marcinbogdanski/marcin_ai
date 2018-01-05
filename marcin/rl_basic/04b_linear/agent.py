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
    def __init__(self, size, step_size=0.1, nb_steps=None, lmbda=None):
        self.V = np.zeros([size])
        self.V[0] = 0   # initialise state-values of terminal states to zero!
        self.V[-1] = 0

        self._step_size = step_size  # usually noted as alpha in literature
        self._discount = 1.0         # usually noted as gamma in literature

        self._nb_steps = nb_steps    # param for nstep_offline function
        self._lmbda = lmbda          # param for lambda functions

        self._episode = 0
        self._trajectory = []        # Agent saves history on it's way
        self._eligibility_traces = {}   # for lambda funtions

    def reset(self):
        self._episode += 1
        self._trajectory = []        # Agent saves history on it's way
        self._eligibility_traces = {}

    def pick_action(self, obs):
        # Randomly go left or right
        return np.random.choice([-1, 1])

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

    def eval_td_t(self, t, V=None):
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

        if V is None:
            V = self.V          # State values array, shape: [world_size]

        # Shortcuts for more compact notation:

        St = self._trajectory[t].observation      # evaluated state tuple (x, y)
        St_1 = self._trajectory[t+1].observation  # next state tuple (x, y)
        Rt_1 = self._trajectory[t+1].reward       # next step reward

        V[St] = V[St] + self._step_size * (Rt_1 + self._discount*V[St_1] - V[St])

    def eval_td_online(self):
        self.eval_td_t(len(self._trajectory) - 2)  # Eval next-to last state

    def eval_td_offline(self, V=None):
        """ Do TD update for all states in trajectory

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
            V = self.V          # State values array, shape: [size]
        
        if not self._trajectory[-1].done:
            raise ValueError('Cant do offline on non-terminated episode')

        # Iterate all states in trajectory
        for t in range(0, len(self._trajectory)-1):
            # Update state-value at time t
            self.eval_td_t(t)

    def calc_Gt(self, t, n=float('inf'), V=None):
        """Calculates return for state t, using n future steps.

        Params:
            t (int [t, T-1]) - time step in trajectory,
                    0 is initial state; T-1 is last non-terminal state

            n (int or +inf, [0, +inf]) - n-steps of reward to accumulate
                    If n >= T then calculate full return for state t
                    For n == 1 this equals to TD return
                    For n == +inf this equals to MC return

            V (float arr) - optional,
                    if passed, funciton will operate on this array
                    if None, then function will operate on self.V    
        """

        if V is None:
            V = self.V          # State values array, shape: [size]

        T = len(self._trajectory)-1   # terminal state
        max_j = min(t+n, T)    # last state iterated, inclusive
        discount = 1.0

        Gt = 0

        # Iterate from t+1 to t+n or T (inclusive on both start and finish)
        #print('calc_GT()  t, n = ', t, n)
        for j in range(t+1, max_j+1):
            Rj = self._trajectory[j].reward
            Gt += discount * Rj
            discount *= self._discount

        # Note that V[Sj] will have state-value of state n+t or
        # zero if n+t >= T as V[St=T] must equal 0
        Sj = self._trajectory[j].observation
        Gt += discount * V[Sj]

        return Gt

    def eval_nstep_t(self, t, V=None):
        """n-step update state-value for single state in trajectory

        This assumesss time steps t+1 to t+n are availalbe in the trajectory

        For online updates:
            Call with t equal to n-1 time step,
            Then at termination, call for each of remaining steps including T-1
        
        For offline updates:
            Iterate trajectory from t=0 to t=T-1 and call for every t

        Params:
            t (int [t, T-1]) - time step in trajectory,
                    0 is initial state; T-1 is last non-terminal state
            
            V (float arr) - optional,
                    if passed, funciton will operate on this array
                    if None, then function will operate on self.V   

        Class Params:
            self._nb_steps (int or +inf, [0, +inf]) - 
                n-steps of reward to account for each state

        """

        if V is None:
            V = self.V          # State values array, shape: [size]
        
        # Shortcuts for more compact notation:
        V = self.V
        St = self._trajectory[t].observation      # evaluated state tuple (x, y)
        Gt = self.calc_Gt(t, n=self._nb_steps, V=V)

        V[St] = V[St] + self._step_size * (Gt - V[St])


    def eval_nstep_offline(self, V=None):
        """n-step update for all steps in trajectory

        Params:            
            V (float arr) - optional,
                    if passed, funciton will operate on this array
                    if None, then function will operate on self.V

        Class Params:
            self._nb_steps (int or +inf, [0, +inf]) - 
                n-steps of reward to account for each state
        """

        if V is None:
            V = self.V          # State values array, shape: [size]
        
        if not self._trajectory[-1].done:
            raise ValueError('Cant do offline on non-terminated episode')

        for t in range(0, len(self._trajectory)-1):
            self.eval_nstep_t(t, self._nb_steps)
        

    def eval_lambda_return_t(self, t, V=None):
        """TD-Lambda update state-value for single state in trajectory

        This assumesss time steps from t+1 to terminating state
        are available in the trajectory

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

        Class Params:
            self._lmbda (float, [0, 1]) - param. for weighted average of returns
        """

        if V is None:
            V = self.V             # State values array, shape: [size]

        # Do offline update only if episode terminated
        if not self._trajectory[-1].done:
            raise ValueError('Cant do offline on non-terminated episode')

        lambda_trunctuate = 1e-3   # discard if lmbda drops below this
        
        # Shortcuts for more compact notation:
        St = self._trajectory[t].observation      # evaluated state tuple (x, y)
        Gt_lambda = 0           # weigthed averated return for current state
        
        T = len(self._trajectory)-1
        max_n = T-t-1           # inclusive

        lmbda_iter = 1
        for n in range(1, max_n+1):
            Gtn = self.calc_Gt(t, n=n, V=V)
            Gt_lambda += lmbda_iter * Gtn
            lmbda_iter *= self._lmbda

            if lmbda_iter < lambda_trunctuate:
                break

        Gt_lambda *= (1 - self._lmbda)

        if lmbda_iter >= lambda_trunctuate:
            Gt_lambda += lmbda_iter * self.calc_Gt(t, V=V)

        V[St] = V[St] + self._step_size * (Gt_lambda - V[St])

        return V

    def eval_lambda_return_offline(self, V=None):
        """Lambda-return update for all states. Inefficient. Use TD-Lambda.

        Note:
            This will perform almost exactly the same update ad TD-Lambda,
            but is much slower. Use only for testing.

        Params:            
            V (float arr) - optional,
                    if passed, funciton will operate on this array
                    if None, then function will operate on self.V

        Class Params:
            self._lmbda (float, [0, 1]) - param. for weighted average of returns

        """

        if V is None:
            V = self.V

        # Do TD-lambda update only if episode terminated
        if self._trajectory[-1].done:

            # Iterate all states in trajectory
            for t in range(0, len(self._trajectory)-1):
                # Update state-value at time t
                self.eval_lambda_return_t(t, V)

        return V

    def eval_td_lambda_t(self, t, V=None):
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

            V (float arr) - optional,
                    if passed, funciton will operate on this array
                    if None, then function will operate on self.V
        """

        if V is None:
            V = self.V

        E = self._eligibility_traces   # eligibility trace dictionary

        St = self._trajectory[t].observation  # current state xy
        St_1 = self._trajectory[t+1].observation
        Rt_1 = self._trajectory[t+1].reward

        if St not in E:
            E[St] = 0

        # Update eligibility traces
        for s in E:
            E[s] *= self._lmbda
        E[St] += 1

        ro_t = Rt_1 + self._discount * V[St_1] - V[St]
        for s in E:
            V[s] = V[s] + self._step_size * ro_t * E[s]

    def eval_td_lambda_offline(self, V=None):
        """TD(lambda) update for all states

        Params:
            V (float arr) - optional,
                    if passed, funciton will operate on this array
                    if None, then function will operate on self.V

        Class Params:
            self._lmbda (float, [0, 1]) - param. for weighted average of returns
        """

        if len(self._eligibility_traces) != 0:
            raise ValueError('TD-lambda offline: eligiblity traces not empty?')

        if V is None:
            V = self.V

        # Do offline update only if episode terminated
        if not self._trajectory[-1].done:
            raise ValueError('Cant do offline on non-terminated episode')

        # Iterate all states apart from terminal state
        max_t = len(self._trajectory)-2  # inclusive
        for t in range(0, max_t+1):
            self.eval_td_lambda_t(t, V=V)

        return V

    def eval_td_lambda_online(self, V=None):
        t = len(self._trajectory) - 2   # Previous time step
        return self.eval_td_lambda_t(t, V)


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

        if V is None:
            V = self.V   # State values array, shape: [size]

        # Shortcuts for more compact notation:
        St = self._trajectory[t].observation  # current state (x, y)
        Gt = self.calc_Gt(t, V=V)            # return for current state

        V[St] = V[St] + self._step_size * (Gt - V[St])

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
        if not self._trajectory[-1].done:
            raise ValueError('Cant do offline on non-terminated episode')

        # Iterate all states in trajectory, apart from terminal state
        for t in range(0, len(self._trajectory)-1):
            # Update state-value at time t
            self.eval_mc_t(t, V)

