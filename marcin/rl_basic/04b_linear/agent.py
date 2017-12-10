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

    def calc_Gtn(self, t, n=float('inf')):
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
        Gt = self.calc_Gtn(t, n)

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
        

    def eval_td_lambda_t(self, t, lmbda):
        """TD-Lambda update state-value for single state in trajectory

        This assumesss time steps from t+1 to terminating state
        are available in the trajectory

        For online updates:
            N/A
        For offline updates:
            Iterate trajectory from t=0 to t=T-1 and call for every t
        """

        #self.print_trajectory()
        #print('Calc TD_lambda for state: t = ', t)

        lambda_trunctuate = 1e-3   # discard if lmbda drops below this
        
        # Shortcuts for more compact notation:
        V = self.V              # State values array, shape: [size_x, size_y]
        St = self.trajectory[t].observation      # evaluated state tuple (x, y)
        Gt_lambda = 0           # weigthed averated return for current state
        
        #print('Handle n loop:')

        T = len(self.trajectory)-1
   
        max_n = T-t-1           # inclusive


        #  just a trivial optimisation, Gt_lambda will be multiplied by
        #  (1 - lmbda) later on, so if lmbda == 1, then no point doing the loop
        lmbda_iter = 1
        for n in range(1, max_n+1):

            #print('  n = ', n)
            #print('  lmbda_pow = ', n-1)
            #print('  lmbda_iter = ', lmbda_iter)

            Gtn = self.calc_Gtn(t, n)
            #print('  Gtn = ', Gtn)
            Gt_lambda += lmbda_iter * Gtn
            lmbda_iter *= lmbda

            if lmbda_iter < lambda_trunctuate:
                break

        Gt_lambda *= (1 - lmbda)

        #print('Handle terminal state, T = ', T)
        #print('  lmbda_pow = ', T-t-1)
        #print('  lmbda_iter = ', lmbda_iter)

        if lmbda_iter >= lambda_trunctuate:
            Gt_lambda += lmbda_iter * self.calc_Gtn(t)

        #print('Gt_lambda = ', Gt_lambda)

        V[St] = V[St] + self.step*(Gt_lambda - V[St])

    def eval_td_lambda_offline(self, lmbda):
        # Do TD-lambda update only if episode terminated
        if self.trajectory[-1].done:

            # Iterate all states in trajectory
            for t in range(0, len(self.trajectory)-1):
                # Update state-value at time t
                self.eval_td_lambda_t(t, lmbda)

        #if self._episode == 1:
        #    print('LB V = ', np.round(self.V[len(self.V)//2-3:],2))

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
        Gt = 0                 # return for current state

        # Iterate all states after this one
        for j in range(t+1, len(self.trajectory)):
            Rj = self.trajectory[j].reward   # reward at time j
            Gt += self.disc**j * Rj          # add with discount

        V[St] = V[St] + self.step*(Gt - V[St])

        #print('MC V = ', np.round(V,2))

    def eval_mc_offline(self):
        # Do MC update only if episode terminated
        if self.trajectory[-1].done:

            # Iterate all states in trajectory
            for t in range(0, len(self.trajectory)-1):
                # Update state-value at time t
                self.eval_mc_t(t)

        #if self._episode == 1:
        #    print('MC V = ', np.round(self.V[len(self.V)//2-3:],2))