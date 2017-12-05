import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import pdb

class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.transitions = {}

        self.is_terminal = False

    def __str__(self):
        return 'St[{0},{1}]'.format(self.x, self.y)


class GridworldEnv:
    def __init__(self, size_x, size_y):
        """
        Indexed from bottm left, like on math graphs
        """
        self.size_x = size_x
        self.size_y = size_y
        self.grid = np.empty( [size_x, size_y], dtype=Cell)

        for x in range(size_x):
            for y in range(size_y):
                self.grid[x, y] = Cell(x=x, y=y)

        for x in range(size_x):
            for y in range(size_y):
                cell = self.grid[x, y]

                # North (0)
                if y == size_y-1:
                    cell.transitions[0] = self.grid[x, y]
                else:
                    cell.transitions[0] = self.grid[x, y+1]

                # East (1)
                if x == size_x-1:
                    cell.transitions[1] = self.grid[x, y]
                else:
                    cell.transitions[1] = self.grid[x+1, y]

                # South (2)
                if y == 0:
                    cell.transitions[2] = self.grid[x, y]
                else:
                    cell.transitions[2] = self.grid[x, y-1]

                # West (3)
                if x == 0:
                    cell.transitions[3] = self.grid[x, y]
                else:
                    cell.transitions[3] = self.grid[x-1, y]


    def print_v(self, heading):
        print(heading)
        for x in range(self.size_x):
            for y in range(self.size_y):
                st = self.world[x][y]
                print('{0: .2f}  '.format(st.v), end='')
            print()

    def plot_world(self, axis, title, V=None, Q=None):
        axis.set_title(title)
        axis.set_xlim([-1,self.size_x])
        axis.set_ylim([-1,self.size_y])
        axis.set_aspect('equal', 'datalim')

        
        for x in range(self.size_x):
            for y in range(self.size_y):
                cell = self.grid[x, y]

                
                axis.add_patch(
                    patches.Rectangle(
                        (cell.x-0.5, cell.y-0.5), 1, 1, fill=False))

                if cell.is_terminal:
                    axis.add_patch(
                        patches.Rectangle(
                            (cell.x-0.5, cell.y-0.5), 1, 1, alpha=0.3))

                # axis.text(st.x-0.45, st.y+0.3, 
                #     '[{0},{1}]'.format(st.x, st.y),
                #     color='lightgrey',
                #     fontsize=6)

                if V is not None:
                    axis.text(cell.x-0.45, cell.y+0.3,
                        '{0:.2f}'.format(V[cell.x, cell.y]), 
                        color='red',
                        horizontalalignment='left',
                        fontsize=6)

                Q_as_arrows = True
                if Q is not None and Q_as_arrows == False:
                    axis.text(cell.x, cell.y+0.25,  # North
                        '{0:.2f}'.format(Q[cell.x, cell.y, 0]), 
                        horizontalalignment='center', fontsize=6)
                    axis.text(cell.x+0.25, cell.y,  # East
                        '{0:.2f}'.format(Q[cell.x, cell.y, 1]), 
                        horizontalalignment='center', fontsize=6)
                    axis.text(cell.x, cell.y-0.25,  # South
                        '{0:.2f}'.format(Q[cell.x, cell.y, 2]), 
                        horizontalalignment='center', fontsize=6)
                    axis.text(cell.x-0.25, cell.y,  # West
                        '{0:.2f}'.format(Q[cell.x, cell.y, 3]), 
                        horizontalalignment='center', fontsize=6)

                if Q is not None and Q_as_arrows == True:

                    # Normalise Q to range 0..1
                    min_q = np.min(Q)
                    max_q = np.max(Q)
                    norm_Q = (Q - min_q) / (max_q - min_q)

                    # Max arrow length
                    arrow_length_mult = 0.35


                    arr_len = norm_Q[cell.x, cell.y, 0] * arrow_length_mult + 1e-3
                    arr_alpha = norm_Q[cell.x, cell.y, 0]
                    axis.arrow(cell.x, cell.y, 0, arr_len,   # North
                        head_width=0.05, head_length=0.1, alpha=arr_alpha, fc='k', ec='k')

                    arr_len = norm_Q[cell.x, cell.y, 1] * arrow_length_mult + 1e-3
                    arr_alpha = norm_Q[cell.x, cell.y, 1]
                    axis.arrow(cell.x, cell.y, arr_len, 0,   # East
                        head_width=0.05, head_length=0.1, alpha=arr_alpha, fc='k', ec='k')

                    arr_len = norm_Q[cell.x, cell.y, 2] * arrow_length_mult + 1e-3
                    arr_alpha = norm_Q[cell.x, cell.y, 2]
                    axis.arrow(cell.x, cell.y, 0, -arr_len,  # South
                        head_width=0.05, head_length=0.1, alpha=arr_alpha, fc='k', ec='k')

                    arr_len = norm_Q[cell.x, cell.y, 3] * arrow_length_mult + 1e-3
                    arr_alpha = norm_Q[cell.x, cell.y, 3]
                    axis.arrow(cell.x, cell.y, -arr_len, 0,  # West
                        head_width=0.05, head_length=0.1, alpha=arr_alpha, fc='k', ec='k')

        
if __name__ == '__main__':
    
    env = GridworldEnv(4, 4)

    fig = plt.figure('Value Iteration')
    ax0 = fig.add_subplot(111)

    env.plot_world(ax0, 'k=0')

    plt.show()

