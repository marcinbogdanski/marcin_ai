import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pdb

def gen_rand_walk():
    state = 500
    states = [state]
    rewards = []

    while True:
        action = np.random.choice([0, 1])

        if action == 0:
            movement = np.random.randint(-100, 0)
        else:
            movement = np.random.randint(1, 101)

        state += movement

        if state > 1000:
            states.append(1000)
            rewards = [1] * len(states)
            break
        elif state < 1:
            states.append(-1)
            rewards = [-1] * len(states)
            break

        states.append(state)

    for i in range(len(states)):
        states[i] /= 1000
        
    return states, rewards

np.random.seed(0)

states = []
rewards = []
for i in range(3000):
    st, rw = gen_rand_walk()
    states += st
    rewards += rw

slope, intercept, _, _, _ = scipy.stats.linregress(states, rewards)
lin_x = np.linspace(0, 1, 1000)
lin_y = slope * lin_x + intercept

mysl = 0
myin = 0

myslopes = [mysl]
myintercepts = [myin]

step_size = 0.0015
for i in range(len(states)):
    state = states[i]
    target = rewards[i]

    est = mysl * state + myin

    # print('------')
    # print('sl, in: ', mysl, myin)
    # print('state:', state)
    # print('target:', target)
    # print('est: ', est)
    
    mysl += step_size * (target-est) * state
    myin += step_size * (target-est)

    myslopes.append(mysl)
    myintercepts.append(myin)

    # print('sl, in: ', mysl, myin)

my_x = np.linspace(0, 1, 1000)
my_y = mysl * my_x + myin
    
print('slope, intercept', slope, intercept)

# print('states:')
# print(states)
# print('rewards:')
# print(rewards)

fig = plt.figure()
ax1 = fig.add_subplot(121)

ax1.plot(lin_x, lin_y, color='gray')
ax1.plot(my_x, my_y, color='green')
ax1.scatter(states, rewards, marker='.')

ax2 = fig.add_subplot(122)
ax2.plot(myslopes, label='myslopes')
ax2.plot(myintercepts, label='myintercepts')

ax2.legend()

plt.show()