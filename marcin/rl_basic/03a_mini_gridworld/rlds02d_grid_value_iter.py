
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import pdb

class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.v = 0
        self.v_temp = 0
        self.actions = []

        self.is_terminal = False

    def __str__(self):
        return 'St[{0},{1}]'.format(self.x, self.y)

class Action:
    def __init__(self, name, prob, reward, target):
        self.name = name
        self.prob = prob
        self.reward = reward
        self.target = target

class GridWorld:
    def __init__(self, size_x, size_y, terminal_st=None):
        """
        Indexed from bottm left, like on math graphs
        """
        self.size_x = size_x
        self.size_y = size_y
        self.terminal_st = terminal_st
        self.world = []

        for x in range(size_x):
            self.world.append([])
            for y in range(size_y):
                st = State(x=x, y=y)
                self.world[x].append(st)

                if terminal_st is not None and (x, y) in terminal_st:
                    st.is_terminal = True

        for x in range(size_x):
            for y in range(size_y):

                
                st = self.world[x][y]
                
                if st.is_terminal:
                    # no getting out from terminal states
                    st.actions.append(Action('T', prob=1.0, reward=0.0, target=st))
                    continue

                # North
                if y == size_y-1:
                    st.actions.append(Action('N', prob=0.25, reward=-1, target=self.world[x][y]))
                else:
                    st.actions.append(Action('N', prob=0.25, reward=-1, target=self.world[x][y+1]))

                # South
                if y == 0:
                    st.actions.append(Action('S', prob=0.25, reward=-1, target=self.world[x][y]))
                else:
                    st.actions.append(Action('S', prob=0.25, reward=-1, target=self.world[x][y-1]))

                # West
                if x == 0:
                    st.actions.append(Action('W', prob=0.25, reward=-1, target=self.world[x][y]))
                else:
                    st.actions.append(Action('W', prob=0.25, reward=-1, target=self.world[x-1][y]))

                # East
                if x == size_x-1:
                    st.actions.append(Action('E', prob=0.25, reward=-1, target=self.world[x][y]))
                else:
                    st.actions.append(Action('E', prob=0.25, reward=-1, target=self.world[x+1][y]))

    def policy_eval_step(self):
        for x in range(self.size_x):
            for y in range(self.size_y):
                st = self.world[x][y]
                st.v_temp = 0
                for a in st.actions:
                    st.v_temp += a.prob * (a.reward + discount * a.target.v)

        for x in range(self.size_x):
            for y in range(self.size_y):
                st = self.world[x][y]
                st.v = st.v_temp                  


    def policy_iter_step(self):
        for x in range(self.size_x):
            for y in range(self.size_y):
                st = self.world[x][y]

                max_v = float('-inf')
                max_a = []
                for a in st.actions:
                    if a.target.v > max_v:
                        max_v = a.target.v
                        max_a.clear()
                        max_a.append(a)
                    elif a.target.v == max_v:
                        max_a.append(a)

                for a in st.actions:
                    if a in max_a:
                        a.prob = 1 / len(max_a)
                    else:
                        a.prob = 0

    def value_iter_step(self):
        for x in range(self.size_x):
            for y in range(self.size_y):
                st = self.world[x][y]

                max_v = float('-inf')
                for a in st.actions:
                    temp_v = a.reward + discount * a.target.v
                    if temp_v > max_v:
                        max_v = temp_v

                st.v = max_v

    def print_v(self, heading):
        print(heading)
        for x in range(self.size_x):
            for y in range(self.size_y):
                st = self.world[x][y]
                print('{0: .2f}  '.format(st.v), end='')
            print()

    def plot_world(self, axis, title):
        axis.set_title(title)
        axis.set_xlim([-1,self.size_x])
        axis.set_ylim([-1,self.size_y])
        
        for x in range(self.size_x):
            for y in range(self.size_y):
                st = self.world[x][y]

                
                axis.add_patch(
                    patches.Rectangle(
                        (st.x-0.5, st.y-0.5), 1, 1, fill=False))

                if st.is_terminal:
                    axis.add_patch(
                        patches.Rectangle(
                            (st.x-0.5, st.y-0.5), 1, 1, alpha=0.3))

                # axis.text(st.x-0.45, st.y+0.3, 
                #     '[{0},{1}]'.format(st.x, st.y),
                #     color='lightgrey',
                #     fontsize=6)

                axis.text(st.x-0.45, st.y+0.3,
                    '{0:.2f}'.format((st.v)), 
                    color='red',
                    horizontalalignment='left',
                    fontsize=6)

                

                max_prob_actions = []
                max_prob = 0

                for a in st.actions:
                    if a.prob > 0:
                        if a.prob > max_prob:
                            max_prob = a.prob
                            max_prob_actions.clear()
                            max_prob_actions.append(a)
                        elif a.prob == max_prob:
                            max_prob_actions.append(a)

                for a in max_prob_actions:
                    if a.name == 'N':
                        axis.arrow(st.x, st.y, 0, 0.3, 
                            head_width=0.05, head_length=0.1, fc='k', ec='k')
                    elif a.name == 'S':
                        axis.arrow(st.x, st.y, 0, -0.3, 
                            head_width=0.05, head_length=0.1, fc='k', ec='k')
                    elif a.name == 'W':
                        axis.arrow(st.x, st.y, -0.3, 0, 
                            head_width=0.05, head_length=0.1, fc='k', ec='k')
                    elif a.name == 'E':
                        axis.arrow(st.x, st.y, 0.3, 0, 
                            head_width=0.05, head_length=0.1, fc='k', ec='k')
        
        
    
world = GridWorld(4, 4, terminal_st=[(0,0),(3,3)])

discount = 1

fig = plt.figure('Value Iteration')
ax0 = fig.add_subplot(231)
ax1 = fig.add_subplot(232)
ax2 = fig.add_subplot(233)
ax3 = fig.add_subplot(234)
ax4 = fig.add_subplot(235)
ax5 = fig.add_subplot(236)

world.plot_world(ax0, 'k=0')

for k in range(1,201):

    #world.policy_eval_step()
    world.value_iter_step()
    world.policy_iter_step()

    # world.print_v('values k=' + str(k))

    if k == 1: world.plot_world(ax1, 'k='+str(k))
    elif k == 2: world.plot_world(ax2, 'k='+str(k))
    elif k == 3: world.plot_world(ax3, 'k='+str(k))
    elif k == 10: world.plot_world(ax4, 'k='+str(k))
    elif k == 200: world.plot_world(ax5, 'k='+str(k))

world.policy_iter_step()


plt.show()

# Time to create 200x200 world: 0.5614066123962402
# Time to evaluate 10x times:   1.351165533065796
