import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from blackjack import BlackjackEnv
import os, sys, pdb

class State:
    def __init__(self, player_points, has_ace, dealer_card):
        self.player_points = player_points
        self.has_ace = has_ace
        self.dealer_card = dealer_card

        self.actions = []

        self.is_terminal = False

        self.value_pi = 0

class Action:
    def __init__(self, name, prob_pi, action):
        self.name = name
        self.action = action

        self.prob_pi = prob_pi

class BlackjackAgent:
    def __init__(self):
        self.alpha = 0.01
        self.states = {}

        self._visited_states_returns = {}

        # deterministic policy: if points below 20 then draw, else stick
        for player_points in range(12, 32):
            for has_ace in [True, False]:
                for dealer_card in range(1,11):
                    st = State(player_points, has_ace, dealer_card)
                    if player_points <= 21:
                        # non terminal states
                        # (21 still requires player to perform action "stick")
                        st.is_terminal = False
                        if player_points < 20:
                            st.actions.append(Action('Draw', 1.0, True))
                        else:
                            st.actions.append(Action('Stick', 1.0, False))
                    else:
                        # terminal states
                        st.is_terminal = True
                    self.states[(player_points, has_ace, dealer_card)] = st

    def reset(self):
        self._visited_states_returns = {}

    def pick_action(self, obs):
        state = self.states[obs]
        action = state.actions[0]
        return action
                

    def learn_mc(self):
        for key in self._visited_states_returns:
            state = self.states[key]
            if state.is_terminal:
                # do not MC update terminal states, duh
                continue

            returnn = self._visited_states_returns[key]
            new_val = \
                state.value_pi + self.alpha * (returnn - state.value_pi)
            state.value_pi = new_val


    def remember(self, obs, reward):
        if obs not in self._visited_states_returns:
            self._visited_states_returns[obs] = 0
        
        for obs in self._visited_states_returns:
            self._visited_states_returns[obs] += reward


if __name__ == '__main__':
    env = BlackjackEnv()
    agent = BlackjackAgent()


    total_ep = 100000
    for episode in range(total_ep):
        if episode % 1000 == 0:
            print('episode: {0} / {1}', episode, total_ep)
        # print()
        # print('  ==  EPISODE {0} START  =='.format(episode))

        obs = env.reset()
        agent.reset()

        # print('HAND:  ', end=''); print(env.player_hand)
        # print('STATE pp={0}, ha={1}, dc={2}'.format(obs[0], obs[1], obs[2]))
        
        while True:

            action = agent.pick_action(obs)

            # print('ACTION: {0}'.format(action.action))

            #   ---   time step rolls here   ---

            obs, reward, done = env.step(action)

            # print('   ------ t={0} -----'.format(env.t_step))
            # print('REWARD: {0}'.format(reward))
            # print('HAND:  ', end=''); print(env.player_hand)
            # print('STATE pp={0}, ha={1}, dc={2}'.format(obs[0], obs[1], obs[2]))
            # print('DONE', done)
            
            agent.remember(obs, reward)


            if done:
                agent.learn_mc()
                break

        # print('  ==  GAME OVER  =='.format(env.t_step))



# no ace plot
player_points = list(range(12, 22))
dealer_card = list(range(1, 11))
X, Y = np.meshgrid(dealer_card, player_points)
Z = np.zeros([len(player_points), len(dealer_card)])

for dc in dealer_card:
    for pp in player_points:
        val = agent.states[(pp, False, dc)].value_pi
        Z[player_points.index(pp), dealer_card.index(dc)] = val

#pdb.set_trace()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, Z)
plt.show()

