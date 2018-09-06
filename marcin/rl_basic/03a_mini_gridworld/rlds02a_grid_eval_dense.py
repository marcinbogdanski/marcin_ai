
import numpy as np


def calc_gridworld_Pss(states, terminal_st=None):
    """Produces state transition probability matrix
            to
         [      ]
    from [      ]
         [      ]
    """

    Pss = np.zeros([states.size, states.size])

    for row in range(states.shape[0]):
        for col in range(states.shape[1]):
            
            if terminal_st is not None and states[row,col] in terminal_st:
                # no getting out from terminal states
                Pss[states[row, col], states[row, col]] = 1
                continue

            # North
            if row == 0:
                Pss[states[row, col], states[row, col]] += 0.25  # loop in place
            else:
                Pss[states[row, col], states[row-1, col]] += 0.25

            # South
            if row == states.shape[0] - 1:
                Pss[states[row, col], states[row, col]] += 0.25
            else:
                Pss[states[row, col], states[row+1, col]] += 0.25

            # West
            if col == 0:
                Pss[states[row, col], states[row, col]] += 0.25
            else:
                Pss[states[row, col], states[row, col-1]] += 0.25

            # East
            if col == states.shape[1] - 1:
                Pss[states[row, col], states[row, col]] += 0.25
            else:
                Pss[states[row, col], states[row, col+1]] += 0.25

    return Pss



states = np.array([[ 0,  1,  2,  3],
                   [ 4,  5,  6,  7],
                   [ 8,  9, 10, 11],
                   [12, 13, 14, 15]])

Pss = calc_gridworld_Pss(states, [0, 15])


Rs = -np.ones([states.size, 1])  # -1 reward everywere
Rs[0]  = 0  # zero reward in terminal states
Rs[-1] = 0



discount = 1

print('states:')
print(states)
print('rewards')
print(Rs.reshape(len(states), -1))

# print('Rs')
# print(Rs)
# print('Pss')
# print(np.array2string(Pss, max_line_width=np.inf))

# state values
V = np.zeros([states.size, 1])

print('values k=0')
print(V.reshape(len(states), -1))

V_temp = np.zeros([states.size, 1])

for k in range(1,11):

    V_temp = Rs + discount * np.dot(Pss, V)
    V = np.copy(V_temp)

    print('values k=', k)
    print(V.reshape(len(states), -1))

