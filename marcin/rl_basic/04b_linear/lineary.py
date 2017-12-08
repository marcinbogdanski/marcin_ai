import numpy as np
import matplotlib.pyplot as plt
import pdb

class LinearEnv:
    """
    Allowed states are:
        [    0         1         2         3         4        ]
           holder             initial              holder
           state               state               state
                 <--       <->       <->       -->
                 -1         0         0         1
    """
    def __init__(self, size):
        self._max_left = 1
        self._max_right = size
        self._start_state = (size // 2) + 1
        self.reset()

    def reset(self):
        self.t_step = 0
        self._state = self._start_state
        self._done = False

        return self._state

    def step(self, action):
        if self._done:
            return (self._state, 0, True)  # We are done

        if action not in [-1, 1]:
            raise ValueError('Invalid action')

        self.t_step += 1
        self._state += action

        obs = self._state
        if self._state > self._max_right:
            reward = +1
            self._done = True
        elif self._state < self._max_left:
            reward = 0
            self._done = True
        else:
            reward = 0
            self._done = False

        return (obs, reward, self._done)

    def print_env(self):
        print('Env state: ', self._state)



class HistoryData:
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
    def __init__(self, size):
        self.V = np.ones([size]) *0.5
        self.V[0] = 0   # initialise state-values of terminal states to zero!
        self.V[-1] = 0

        self.step = 0.1   # step-size parameter - usually noted as alpha
        self.disc = 1.0    # discount factor - usually noted as gamma

        self.reset()

    def reset(self):
        self.trajectory = []

    def pick_action(self, obs):
        # Randomly go left or right
        return np.random.choice([-1, 1])

    def append_trajectory(self, t_step, prev_action, observation, reward, done):
        if len(self.trajectory) != 0:
            self.trajectory[-1].action = prev_action

        self.trajectory.append(
            HistoryData(t_step, obs, reward, done))

    def print_trajectory(self):
        print('Trajectory:')
        for element in self.trajectory:
            print(element)
        print('Total trajectory steps: {0}'.format(len(self.trajectory)))

    def eval_V_td(self):
        # Shortcuts for more compact notation:
        V = self.V              # State values array, shape: [size_x, size_y]
        St = self.trajectory[-2].observation    # previous state tuple (x, y)
        St_1 = self.trajectory[-1].observation  # current state tuple (x, y)
        Rt_1 = self.trajectory[-1].reward       # current step reward

        #print('V', V)
        #print('St', St)
        #print('St_1', St_1)
        #print('Rt_1', Rt_1)

        #print('self.disc*V[St_1]', self.disc*V[St_1])
        #print('Rt_1 + self.disc*V[St_1] - V[St]', Rt_1 + self.disc*V[St_1] - V[St])
        V[St] = V[St] + self.step*(Rt_1 + self.disc*V[St_1] - V[St])

        #print('V', V)
        #print('-----')


if __name__ == '__main__':

    outer_RMS = []

    max_runs = 100

    for run in range(max_runs):


        env = LinearEnv(5)
        true_vals = [0, 1/6, 2/6, 3/6, 4/6, 5/6, 0]
        agent = Agent(5+2)

        RMS = []    # root mean-squared error
        max_episodes = 100
        for e in range(max_episodes):

            # pdb.set_trace()

            obs = env.reset()
            agent.reset()
            agent.append_trajectory(t_step=0,
                                    prev_action=None,
                                    observation=obs,
                                    reward=None,
                                    done=None)
            #rms = np.sqrt(np.sum(np.power(true_vals - agent.V, 2)))
            #RMS.append(rms)

            #pdb.set_trace()

            while True:

                action = agent.pick_action(obs)

                #   ---   time step rolls here   ---

                obs, reward, done = env.step(action)

                agent.append_trajectory(t_step=env.t_step,
                            prev_action=action,
                            observation=obs,
                            reward=reward,
                            done=done)

                agent.eval_V_td()

                if done:
                    break

            rms = np.sqrt(np.sum(np.power(true_vals - agent.V, 2)) / 5)
            RMS.append(rms)

        outer_RMS.append(RMS)
    
    outer_RMS = np.array(outer_RMS)
    average_RMS = np.sum(outer_RMS, axis=0) / len(outer_RMS)


    agent.print_trajectory()

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(true_vals)
    ax.plot(agent.V, label='TD')
    
    ax = fig.add_subplot(122)

    #for i in range(len(outer_RMS)):
    ax.plot(average_RMS)

    plt.grid()
    plt.show()

    print(agent.V)