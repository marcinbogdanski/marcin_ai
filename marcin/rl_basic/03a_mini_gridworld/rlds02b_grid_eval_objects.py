
import numpy as np
import time

class State:
    def __init__(self, identifier):
        self.id = identifier
        self.v = 0
        self.v_temp = 0
        self.actions = []

class Action:
    def __init__(self, prob, reward, target):
        self.prob = prob
        self.reward = reward
        self.target = target

def create_gridworld(size_x, size_y, terminal_st=None):
    """
    Indexed from bottm left, like on math graphs
    """

    world = []

    for x in range(size_x):
        world.append([])
        for y in range(size_y):
            st = State(identifier=[x,y])
            world[x].append(st)

    for x in range(size_x):
        for y in range(size_y):

            coord = (x, y)
            st = world[x][y]
            
            if terminal_st is not None and coord in terminal_st:
                # no getting out from terminal states
                st.actions.append(Action(prob=1.0, reward=0.0, target=st))
                continue

            # North
            if y == size_y-1:
                st.actions.append(Action(prob=0.25, reward=-1, target=world[x][y]))
            else:
                st.actions.append(Action(prob=0.25, reward=-1, target=world[x][y+1]))

            # South
            if y == 0:
                st.actions.append(Action(prob=0.25, reward=-1, target=world[x][y]))
            else:
                st.actions.append(Action(prob=0.25, reward=-1, target=world[x][y-1]))

            # West
            if x == 0:
                st.actions.append(Action(prob=0.25, reward=-1, target=world[x][y]))
            else:
                st.actions.append(Action(prob=0.25, reward=-1, target=world[x-1][y]))

            # East
            if x == size_x-1:
                st.actions.append(Action(prob=0.25, reward=-1, target=world[x][y]))
            else:
                st.actions.append(Action(prob=0.25, reward=-1, target=world[x+1][y]))

    return world

def print_v(heading):
    print(heading)
    for x in range(len(world)):
        for y in range(len(world[x])):
            st = world[x][y]
            print('{0: .2f}  '.format(st.v), end='')
        print()

def eval_v_step():
    for x in range(len(world)):
        for y in range(len(world[x])):
            st = world[x][y]
            st.v_temp = 0
            for a in st.actions:
                st.v_temp += a.prob * (a.reward + discount * a.target.v)

    for x in range(len(world)):
        for y in range(len(world[x])):
            st = world[x][y]
            st.v = st.v_temp
    
world = create_gridworld(4, 4, terminal_st=[(0,0),(3,3)])

# for x in range(len(world)):
#     for y in range(len(world[x])):
#         st = world[x][y]
#         print(x, y, st.id)
#         for a in st.actions:
#             print('  ', a.target.id)

discount = 1


print_v('values k=0:')

for k in range(1,11):

    eval_v_step()

    print_v('values k=' + str(k))
    


# Time to create 200x200 world: 0.5614066123962402
# Time to evaluate 10x times:   1.351165533065796
