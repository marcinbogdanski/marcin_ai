
import numpy as np
import scipy.sparse


def calc_gridworld_Pss(states, terminal_st=None):
    """Produces state transition probability matrix
            to
         [      ]
    from [      ]
         [      ]
    """

    Pss = scipy.sparse.dok_matrix((states.size, states.size))

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

nb_cols = 4
nb_rows = 4

states = np.array(range(nb_cols*nb_rows)).reshape(nb_rows,-1)

Pss = calc_gridworld_Pss(states, [0, states.size-1])

Rs = -np.ones([states.size, 1])  # -1 reward everywere
Rs[0]  = 0  # zero reward in terminal states
Rs[-1] = 0


# print('states:')
# print(states)
# print('rewards')
# print(Rs.reshape(len(states), -1))

# print('Rs')
# print(Rs)
# print('Pss')
# print(Pss)

# state values
V = np.zeros([states.size, 1])

print('values k=0')
print(V.reshape(len(states), -1))

V_temp = np.zeros([states.size, 1])

discount = 1



for k in range(1,11):

    V_temp = Rs + discount * Pss.dot(V)
    V = np.copy(V_temp)

    print('values k=', k)
    print(V.reshape(len(states), -1))



# Time to create 200x200 world: 9.02400541305542
# Time to evaluate 10x times:   0.978778600692749