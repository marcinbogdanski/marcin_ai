import unittest
import os
import random
import numpy as np
import tensorflow as tf

from logger import Logger, Log
import main

class TestMain(unittest.TestCase):

    def setUp(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # disable GPU
        os.environ['PYTHONHASHSEED'] = '0'         # force reproducible hasing

        #
        #   Random seeds
        #
        self.seed = 1
        print('Using random seed:', self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        #
        #   TF session
        #
        config = tf.ConfigProto()    
        config.intra_op_parallelism_threads=1
        config.inter_op_parallelism_threads=1

        config.gpu_options.per_process_gpu_memory_fraction=0.2
        self.sess = tf.Session(config=config)

        #
        #   Logger etc
        #
        self.logger = Logger('curr_datetime', 'hostname', 'git_hash')
        self.logger.agent = Log('Agent')
        self.logger.q_val = Log('Q_Val')
        self.logger.env = Log('Environment', 'Mountain Car')
        self.logger.hist = Log('History', 'History of all states visited')
        self.logger.memory = Log('Memory', 'Agent full memory dump on given timestep')
        self.logger.approx = Log('Approx', 'Approximator')

        
    def tearDown(self):
        print('teardown')

    def test_run_keras_1(self):
        print('test run 1')

        timing_arr = []
        timing_dict = {}

        trained_agent = main.test_run(
            nb_episodes=None,
            nb_total_steps=11000,
            expl_start=False,

            agent_nb_actions=3,
            agent_discount=0.99,
            agent_nb_rand_steps=10000,
            agent_e_rand_start=1.0,
            agent_e_rand_target=0.1,
            agent_e_rand_decay=1/10000,

            mem_size_max=10000,
            mem_enable_pmr=False,

            approximator='keras',
            step_size=0.00025,
            batch_size=64,
            
            plotter=None,
            logger=self.logger,
            timing_arr=timing_arr,
            timing_dict=timing_dict,
            seed=self.seed)

        fp, ws, st, act, rew, done = trained_agent.get_fingerprint()
        print('FINGERPRINT:', fp)
        print('  wegight sum:', ws)
        print('  st, act, rew, done:', st, act, rew, done)

        self.assertEqual(fp, -2253.8704068369316)
        self.assertEqual(ws, 3396.6942402124405)
        self.assertEqual(st, -5640.564647049372)
        self.assertEqual(act, 10988)
        self.assertEqual(rew, -10999)
        self.assertEqual(done, 1)
