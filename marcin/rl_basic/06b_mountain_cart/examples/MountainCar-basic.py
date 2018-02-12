# OpenGym MountainCar-v0
# -------------------
#
# This code demonstrates debugging of a basic Q-network (without target network)
# in an OpenGym MountainCar-v0 environment.
#
# Made as part of blog series Let's make a DQN, available at: 
# https://jaromiru.com/2016/10/12/lets-make-a-dqn-debugging/
# 
# author: Jaromir Janisch, 2016


#--- enable this to run on GPU
# import os    
# os.environ['THEANO_FLAGS'] = "device=gpu,floatX=float32"  

import random, numpy, math, gym

#-------------------- UTILITIES -----------------------
import matplotlib.pyplot as plt
from matplotlib import colors
import sys
import pdb

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.2
# config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

def printQ(agent):
    P = [
        [-0.15955113,  0.        ], # s_start

        [ 0.83600049,  0.27574312], # s'' -> s'
        [ 0.85796947,  0.28245832], # s' -> s
        [ 0.88062271,  0.29125591], # s -> terminal
    ]

    pred = agent.brain.predict( numpy.array(P) )

    for o in pred:
        sys.stdout.write(str(o[1])+" ")

    print(";", agent.steps)
    sys.stdout.flush()

def mapBrain(brain, res):
    s = numpy.zeros( (res * res, 2) )
    i = 0

    for i1 in range(res):
        for i2 in range(res):            
            s[i] = numpy.array( [ 2 * (i1 - res / 2) / res, 2 * (i2 - res / 2) / res ] )
            i += 1

    mapV = numpy.amax(brain.predict(s), axis=1).reshape( (res, res) )
    mapA = numpy.argmax(brain.predict(s), axis=1).reshape( (res, res) )

    return (mapV, mapA)

fig = None
ax1 = None
ax2 = None
def displayBrain(brain, res=50):    
    global fig, ax1, ax2
    mapV, mapA = mapBrain(brain, res)

    # plt.close()
    # plt.show()  

    if fig is None:
        fig = plt.figure(figsize=(5,7))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
    else:
        ax1.clear()
        ax2.clear()
        # plt.colorbar(orientation='vertical')

    ax1.imshow(mapV)


    #  fig.add_subplot(212)

    cmap = colors.ListedColormap(['blue', 'red'])
    bounds=[-0.5,0.5,1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax2.imshow(mapA, cmap=cmap, norm=norm)        
    # cb = plt.colorbar(orientation='vertical', ticks=[0,1])

    plt.pause(0.001)

#-------------------- BRAIN ---------------------------

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        # self.model.load_weights("MountainCar-basic.h5")

    def _createModel(self):
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Dense(units=256, activation='relu', input_dim=stateCnt))
        model.add(tf.keras.layers.Dense(units=256, activation='relu'))
        model.add(tf.keras.layers.Dense(units=actionCnt, activation='linear'))

        opt = tf.keras.optimizers.RMSprop(lr=0.00025)
        model.compile(loss='mse', optimizer=opt)

        return model

    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=64, epochs=epoch, verbose=verbose)

    def predict(self, s):
        return self.model.predict(s)

    def predictOne(self, s):
        return self.predict(s.reshape(1, self.stateCnt)).flatten()

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)        

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def isFull(self):
        return len(self.samples) >= self.capacity

#-------------------- AGENT ---------------------------
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.1
LAMBDA = 0.001      # speed of decay

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self._random = random.Random()

        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)
        
    def act(self, s):
        if self._random.random() < self.epsilon:
            res = self._random.randint(0, self.actionCnt-1)
        else:
            res = numpy.argmax(self.brain.predictOne(s))
        return res

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)        

        # ----- debug
        if self.steps % 1000 == 0:
            printQ(self)

        if self.steps % 1000 == 0:
            displayBrain(self.brain)

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):    
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([ o[0] for o in batch ])
        states_ = numpy.array([ (no_state if o[3] is None else o[3]) for o in batch ])

        p = agent.brain.predict(states)
        p_ = agent.brain.predict(states_)

        x = numpy.zeros((batchLen, self.stateCnt))
        y = numpy.zeros((batchLen, self.actionCnt))
        
        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * numpy.amax(p_[i])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)

class RandomAgent:
    memory = Memory(MEMORY_CAPACITY)

    def __init__(self, actionCnt):
        self.actionCnt = actionCnt
        self._random = random.Random()
        self._random.seed(0)

    def act(self, s):
        result = self._random.randint(0, self.actionCnt-1)
        return result

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

    def replay(self):
        pass

#-------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem).env
        self.env.seed(0)

        high = self.env.observation_space.high
        low = self.env.observation_space.low

        self.mean = (high + low) / 2
        self.spread = abs(high - low) / 2

        self.total_step = -1

    def normalize(self, s):
        return (s - self.mean) / self.spread

    def run(self, agent):

        self.total_step += 1
        print('  -----------------------    total_step', self.total_step)
        s = self.env.reset()
        print('STEP', s)
        s = self.normalize(s)
        R = 0 

        steps = 0
        while True:            

            # self.env.render()

            a = agent.act(s)    # map actions; 0 = left, 2 = right              
            print('ACTION', a)
            if a == 0: 
                a_ = 0
            elif a == 1: 
                a_ = 2


            if self.total_step >= 100010:
                exit(0)

            # time step roll here

            steps += 1
            self.total_step += 1
            print('  ------------------------    total_step', self.total_step)

            s_, r, done, info = self.env.step(a_)
            print('STEP', s_)
            s_ = self.normalize(s_)

            if done: # terminal state
                s_ = None

            agent.observe( (s, a, r, s_) )
            agent.replay()            

            s = s_
            R += r

            

            
            if done:
                print('terminated after:', steps)
                break
            if isinstance(agent, RandomAgent) and agent.memory.isFull():
                print('   MEMORY FULL   ')
                break

        # print("Total reward:", R)

#-------------------- MAIN ----------------------------
PROBLEM = 'MountainCar-v0'
env = Environment(PROBLEM)

stateCnt  = env.env.observation_space.shape[0]
actionCnt = 2 #env.env.action_space.n

agent = Agent(stateCnt, actionCnt)
randomAgent = RandomAgent(actionCnt)

try:
    while randomAgent.memory.isFull() == False:
        env.run(randomAgent)

    agent.memory = randomAgent.memory
    randomAgent = None

    while True:
        env.run(agent)
finally:
    agent.brain.model.save("MountainCar-basic.h5")
