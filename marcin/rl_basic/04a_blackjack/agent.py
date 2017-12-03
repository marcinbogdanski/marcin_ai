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

        for player_points in range(12, 22):
            for has_ace in [True, False]:
                for dealer_card in range(1,11):
                    st = State(player_points, has_ace, dealer_card)
                    if player_points < 20:
                        st.actions.append(Action('Draw', 1.0, True))
                    else:
                        st.actions.append(Action('Stick', 1.0, False))
                    self.states[(player_points, has_ace, dealer_card)] = st

    def reset_episode(self):
        for key in self._visited_states_returns:
            state = self.states[key]
            returnn = self._visited_states_returns[key]
            new_val = \
                state.value_pi + self.alpha * (returnn - state.value_pi)
            state.value_pi = new_val

        self._visited_states_returns = {}

    def learn(self, player_points, has_ace, dealer_card, reward):
        key = (player_points, has_ace, dealer_card)
        if key not in self._visited_states_returns:
            self._visited_states_returns[key] = 0
        
        for key in self._visited_states_returns:
            self._visited_states_returns[key] += reward


if __name__ == '__main__':
    env = BlackjackEnv()
    agent = BlackjackAgent()

    total_ep = 10000
    for episode in range(total_ep):
        if episode % 100 == 0:
            print('episode: {0} / {1}', episode, total_ep)
        # print()
        # print('  ==  EPISODE {0} START  =='.format(episode))

        obs, reward, done = env.reset()
        player_points = obs[0]
        has_ace = obs[1]
        dealer_card = obs[2]
        agent.learn(player_points, has_ace, dealer_card, reward)
        
        while True:
            player_points = obs[0]
            has_ace = obs[1]
            dealer_card = obs[2]
            
            # print('  ----- t = {0} -----'.format(env.t_step))
            # print('REWARD: {0}'.format(reward))
            # print('HAND:  ', end=''); print(env.player_hand)
            # print('STATE pp={0}, ha={1}, dc={2}'.format(
            #     player_points, has_ace, dealer_card))


            if not done:
                state = agent.states[(player_points, has_ace, dealer_card)]
                action = state.actions[0]
                # print('ACTION: {0}'.format(action.name))

                obs, reward, done = env.step(action.action)

                #################
                #   LEARN
                #
                agent.learn(player_points, has_ace, dealer_card, reward)

                # print(agent._visited_states_returns)
            else:
                break

        agent.reset_episode()
        # print('  ==  GAME OVER  =='.format(env.t_step))



# no ace plot
player_points = list(range(12, 22))
dealer_card = list(range(1, 11))
X, Y = np.meshgrid(dealer_card, player_points)
Z = np.zeros([len(dealer_card), len(player_points)])

for dc in dealer_card:
    for pp in player_points:
        val = agent.states[(pp, False, dc)].value_pi
        Z[player_points.index(pp), dealer_card.index(dc)] = val

#pdb.set_trace()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, Z)
plt.show()

