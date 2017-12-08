import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import pdb

class Cell:
    def __init__(self, x, y, reward=-1):
        self.x = x
        self.y = y
        self.reward = reward
        self.transitions = {}
        
        self.type = 'default'

        self._is_terminal = False

        # For painting:
        self._text_symbol = None
        self._paint_color = None

    def __str__(self):
        return 'St[{0},{1}]'.format(self.x, self.y)

    def make_start(self):
        self.type = 'start'

        self._paint_color = 'black'
        self._text_symbol = 'S'

    def make_goal(self):
        self.type = 'goal'
        self.reward = 0

        self._is_terminal = True

        self._paint_color = 'green'
        self._text_symbol = 'G'

    def make_wall(self):
        self.type = 'wall'

        self._paint_color = 'black'
        self._text_symbol  ='W'



class GridworldEnv:
    def __init__(self, size_x, size_y):
        """
        Indexed from bottm left, like on math graphs
        """
        self.t_step = 0
        self.size_x = size_x
        self.size_y = size_y
        self._grid = np.empty( [size_x, size_y], dtype=Cell)
        self._start_cells = []

        self._player_cell = None
        self._finished = False

        for x in range(size_x):
            for y in range(size_y):
                self._grid[x, y] = Cell(x=x, y=y)

        for x in range(size_x):
            for y in range(size_y):
                cell = self._grid[x, y]

                # North (0)
                if y == size_y-1:
                    cell.transitions[0] = self._grid[x, y]
                else:
                    cell.transitions[0] = self._grid[x, y+1]

                # East (1)
                if x == size_x-1:
                    cell.transitions[1] = self._grid[x, y]
                else:
                    cell.transitions[1] = self._grid[x+1, y]

                # South (2)
                if y == 0:
                    cell.transitions[2] = self._grid[x, y]
                else:
                    cell.transitions[2] = self._grid[x, y-1]

                # West (3)
                if x == 0:
                    cell.transitions[3] = self._grid[x, y]
                else:
                    cell.transitions[3] = self._grid[x-1, y]

    def make_start(self, x, y):
        cell = self._grid[x, y]
        cell.make_start()

        if cell not in self._start_cells:
            self._start_cells.append(cell)

    def make_goal(self, x, y):
        cell = self._grid[x, y]
        cell.make_goal()

        for key in cell.transitions:
            cell.transitions[key] = cell

    def make_wall(self, x, y):
        cell = self._grid[x, y]
        cell.make_wall()

        for key in cell.transitions:
            cell.transitions[key] = cell

        if y < self.size_y-1:
            cell_N = self._grid[x, y+1]
            cell_N.transitions[2] = cell_N

        if x < self.size_x-1:
            cell_W = self._grid[x+1, y]
            cell_W.transitions[3] = cell_W

        if y > 0:
            cell_S = self._grid[x, y-1]
            cell_S.transitions[0] = cell_S

        if x > 0:
            cell_E = self._grid[x-1, y]
            cell_E.transitions[1] = cell_E

    def reset(self):
        self.t_step = 0
        rnd_idx = np.random.randint(0, len(self._start_cells))
        cell = self._start_cells[rnd_idx]
        self._player_cell = cell
        self._finished = False

        obs = (cell.x, cell.y)

        return obs

    def step(self, action):

        # TODO: if in terminal state, then stop incrementing time step?

        self.t_step += 1
        curr_cell = self._player_cell
        new_cell = curr_cell.transitions[action]
        self.player_cell = new_cell
        self._finished = new_cell._is_terminal

        self._player_cell = new_cell

        obs = (new_cell.x, new_cell.y)
        reward = curr_cell.reward
        done = self._finished

        return (obs, reward, done)


    def print_v(self, heading):
        print(heading)
        for x in range(self.size_x):
            for y in range(self.size_y):
                st = self.world[x][y]
                print('{0: .2f}  '.format(st.v), end='')
            print()

    def plot_world(self, axis, title, V=None, Q=None, 
        trajectory=None,
        plot_transitions=False, 
        plot_rewards=False):
        axis.clear()
        axis.set_title(title)
        axis.set_xlim([-1,self.size_x])
        axis.set_ylim([-1,self.size_y])
        axis.set_aspect('equal', 'datalim')
        
        for x in range(self.size_x):
            for y in range(self.size_y):
                cell = self._grid[x, y]

                ########################################
                # Draw square borders,
                # player occupied cell has bold border
                if self._player_cell is not None and cell == self._player_cell:
                    axis.add_patch(
                        patches.Rectangle(
                            (cell.x-0.5, cell.y-0.5), 1, 1, 
                            fill=False, linewidth=3))
                else:
                    axis.add_patch(
                        patches.Rectangle(
                            (cell.x-0.5, cell.y-0.5), 1, 1, fill=False))
                                   
                ########################################
                # Cell type
                # Top-left corder reserved for letter indicating cell type
                if cell._paint_color is not None:
                    axis.add_patch(
                        patches.Rectangle(
                            (cell.x-0.5, cell.y-0.5), 1, 1,
                            alpha=0.3, color=cell._paint_color))

                if cell._text_symbol is not None:
                    axis.text(cell.x-0.4, cell.y+0.2, 
                        cell._text_symbol,
                        color=cell._paint_color,
                        fontsize=10)

                ########################################
                # Plot trajectory
                if trajectory is not None:
                    for i in range(1, len(trajectory)):
                        st = trajectory[i-1]
                        en = trajectory[i]
                        axis.arrow(st[0]+0.15, st[1]+0.15,
                            en[0]-st[0], en[1]-st[1], 
                            head_width=0.05, head_length=0.1,
                            fc='blue', ec='blue')

                ########################################
                # Plot transitions
                if plot_transitions:
                    for key, val in cell.transitions.items():
                        if cell != val:
                            axis.arrow(cell.x, cell.y, 
                                (val.x - cell.x) / 3,
                                (val.y - cell.y) / 3,
                                head_width=0.05, head_length=0.1, 
                                fc='black', ec='black')

                ########################################
                # Plot rewards
                # Top-right cornder reserved for rewards
                if plot_rewards:
                    axis.text(cell.x+0.4, cell.y+0.2, 
                        '{0:d}'.format(round(cell.reward)),
                        horizontalalignment='right',
                        color='black',
                        fontsize=10)


                ########################################
                # Plot state-value function
                # Bottom-left reserved for state-value function
                if V is not None:
                    axis.text(cell.x-0.45, cell.y-0.3,
                        '{0:.2f}'.format(V[cell.x, cell.y]), 
                        color='red',
                        horizontalalignment='left',
                        fontsize=6)
               

                ########################################
                # Plot action-value funciton
                # Text drawn on-top-of arrows
                Q_as_arrows = False
                if Q is not None and Q_as_arrows == False:
                    axis.text(cell.x, cell.y+0.25,  # North
                        '{0:.2f}'.format(Q[cell.x, cell.y, 0]), 
                        horizontalalignment='center', fontsize=6, color='red')
                    axis.text(cell.x+0.25, cell.y,  # East
                        '{0:.2f}'.format(Q[cell.x, cell.y, 1]), 
                        horizontalalignment='center', fontsize=6, color='red')
                    axis.text(cell.x, cell.y-0.25,  # South
                        '{0:.2f}'.format(Q[cell.x, cell.y, 2]), 
                        horizontalalignment='center', fontsize=6, color='red')
                    axis.text(cell.x-0.25, cell.y,  # West
                        '{0:.2f}'.format(Q[cell.x, cell.y, 3]), 
                        horizontalalignment='center', fontsize=6, color='red')

                if Q is not None and Q_as_arrows == True:

                    # Normalise Q to range 0..1
                    min_q = np.min(Q)
                    max_q = np.max(Q)
                    norm_Q = (Q - min_q) / (max_q - min_q)

                    # Max arrow length
                    arrow_scale = 0.35
                    arrow_color = 'red'

                    # zero-length arrows point upward ackwardly, so we add 1e-3
                    arr_len = norm_Q[cell.x, cell.y, 0] * arrow_scale + 1e-3
                    arr_alpha = norm_Q[cell.x, cell.y, 0]
                    axis.arrow(cell.x, cell.y, 0, arr_len,   # North
                        head_width=0.05, head_length=0.1, alpha=arr_alpha,
                        fc=arrow_color, ec=arrow_color)

                    arr_len = norm_Q[cell.x, cell.y, 1] * arrow_scale + 1e-3
                    arr_alpha = norm_Q[cell.x, cell.y, 1]
                    axis.arrow(cell.x, cell.y, arr_len, 0,   # East
                        head_width=0.05, head_length=0.1, alpha=arr_alpha,
                        fc=arrow_color, ec=arrow_color)

                    arr_len = norm_Q[cell.x, cell.y, 2] * arrow_scale + 1e-3
                    arr_alpha = norm_Q[cell.x, cell.y, 2]
                    axis.arrow(cell.x, cell.y, 0, -arr_len,  # South
                        head_width=0.05, head_length=0.1, alpha=arr_alpha,
                        fc=arrow_color, ec=arrow_color)

                    arr_len = norm_Q[cell.x, cell.y, 3] * arrow_scale + 1e-3
                    arr_alpha = norm_Q[cell.x, cell.y, 3]
                    axis.arrow(cell.x, cell.y, -arr_len, 0,  # West
                        head_width=0.05, head_length=0.1, alpha=arr_alpha,
                        fc=arrow_color, ec=arrow_color)

        
if __name__ == '__main__':
    
    env = GridworldEnv(4, 4)

    fig = plt.figure('Value Iteration')
    ax0 = fig.add_subplot(111)

    env.plot_world(ax0, 'k=0')

    plt.show()

