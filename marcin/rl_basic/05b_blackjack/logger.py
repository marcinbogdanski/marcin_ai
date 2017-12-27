
import numpy as np
import pickle
import pdb

PLAYER_SUM_MIN = 12   # BlackjackEnv guarantees this
PLAYER_SUM_MAX = 31   # 21 + draw 10
DEALER_CARD_MIN = 1   # ace
DEALER_CARD_MAX = 10


class Logger:
    def __init__(self, reference_data_filename=None):
        if reference_data_filename is not None:
            dataset = np.load(reference_data_filename)
            self.ref_Q_no_ace_hold = dataset[:,:,0,1]
            self.ref_Q_no_ace_draw = dataset[:,:,0,0]
            self.ref_Q_ace_hold = dataset[:,:,1,1]
            self.ref_Q_ace_draw = dataset[:,:,1,0]

        self.log_t = []
        self.log_Q_no_ace_hold = []
        self.log_Q_no_ace_draw = []
        self.log_Q_ace_hold = []
        self.log_Q_ace_draw = []

        self.log_Q_num_no_ace_hold = np.zeros([10, 10])
        self.log_Q_num_no_ace_draw = np.zeros([10, 10])
        self.log_Q_num_ace_hold = np.zeros([10, 10])
        self.log_Q_num_ace_draw = np.zeros([10, 10])

    def log(self, episode, V, Q, Q_num):
        Q_no_ace_hold = np.zeros([10, 10])
        Q_no_ace_draw = np.zeros([10, 10])
        Q_ace_hold = np.zeros([10, 10])
        Q_ace_draw = np.zeros([10, 10])
        
        for player_sum in range(PLAYER_SUM_MIN, 21+1):
            for dealer_card in range(DEALER_CARD_MIN, DEALER_CARD_MAX+1):
                Q_no_ace_hold[player_sum-12, dealer_card-1] = \
                    Q[(0, player_sum, dealer_card), 0]
                Q_no_ace_draw[player_sum-12, dealer_card-1] = \
                    Q[(0, player_sum, dealer_card), 1]
                Q_ace_hold[player_sum-12, dealer_card-1] = \
                    Q[(1, player_sum, dealer_card), 0]
                Q_ace_draw[player_sum-12, dealer_card-1] = \
                    Q[(1, player_sum, dealer_card), 1]

                self.log_Q_num_no_ace_hold[player_sum-12, dealer_card-1] = \
                    Q_num[(0, player_sum, dealer_card), 0]
                self.log_Q_num_no_ace_draw[player_sum-12, dealer_card-1] = \
                    Q_num[(0, player_sum, dealer_card), 1]
                self.log_Q_num_ace_hold[player_sum-12, dealer_card-1] = \
                    Q_num[(1, player_sum, dealer_card), 0]
                self.log_Q_num_ace_draw[player_sum-12, dealer_card-1] = \
                    Q_num[(1, player_sum, dealer_card), 1]

        self.log_t.append(episode)
        self.log_Q_no_ace_hold.append(Q_no_ace_hold)
        self.log_Q_no_ace_draw.append(Q_no_ace_draw)
        self.log_Q_ace_hold.append(Q_ace_hold)
        self.log_Q_ace_draw.append(Q_ace_draw)

    def save(self, filename):
        self.log_t = np.array(self.log_t)
        self.log_Q_no_ace_hold = np.array(self.log_Q_no_ace_hold)
        self.log_Q_no_ace_draw = np.array(self.log_Q_no_ace_draw)
        self.log_Q_ace_hold = np.array(self.log_Q_ace_hold)
        self.log_Q_ace_draw = np.array(self.log_Q_ace_draw)

        with open(filename, 'wb') as f:
            pickle.dump( self.__dict__, f )

    def load(self, filename):
        with open(filename, 'rb') as f:
            tmp_dict = pickle.load(f)
            self.__dict__.clear()
            self.__dict__.update(tmp_dict)

