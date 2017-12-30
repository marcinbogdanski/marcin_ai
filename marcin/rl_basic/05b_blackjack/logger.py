
import numpy as np
import pickle
import pdb

PLAYER_SUM_MIN = 12   # BlackjackEnv guarantees this
PLAYER_SUM_MAX = 31   # 21 + draw 10
DEALER_CARD_MIN = 1   # ace
DEALER_CARD_MAX = 10

class DataReference:
    def __init__(self, reference_data_filename):
        dataset = np.load(reference_data_filename)
        self.Q_no_ace_hold = dataset[:,:,0,1]
        self.Q_no_ace_draw = dataset[:,:,0,0]
        self.Q_ace_hold = dataset[:,:,1,1]
        self.Q_ace_draw = dataset[:,:,1,0]

class DataLogger:
    def __init__(self):

        self.t = []
        self.Q_no_ace_hold = []
        self.Q_no_ace_draw = []
        self.Q_ace_hold = []
        self.Q_ace_draw = []

        self.Q_num_no_ace_hold = np.zeros([10, 10])
        self.Q_num_no_ace_draw = np.zeros([10, 10])
        self.Q_num_ace_hold = np.zeros([10, 10])
        self.Q_num_ace_draw = np.zeros([10, 10])

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

                self.Q_num_no_ace_hold[player_sum-12, dealer_card-1] = \
                    Q_num[(0, player_sum, dealer_card), 0]
                self.Q_num_no_ace_draw[player_sum-12, dealer_card-1] = \
                    Q_num[(0, player_sum, dealer_card), 1]
                self.Q_num_ace_hold[player_sum-12, dealer_card-1] = \
                    Q_num[(1, player_sum, dealer_card), 0]
                self.Q_num_ace_draw[player_sum-12, dealer_card-1] = \
                    Q_num[(1, player_sum, dealer_card), 1]

        self.t.append(episode)
        self.Q_no_ace_hold.append(Q_no_ace_hold)
        self.Q_no_ace_draw.append(Q_no_ace_draw)
        self.Q_ace_hold.append(Q_ace_hold)
        self.Q_ace_draw.append(Q_ace_draw)

    def prep_to_save(self):
        self.t = np.array(self.t)
        self.Q_no_ace_hold = np.array(self.Q_no_ace_hold)
        self.Q_no_ace_draw = np.array(self.Q_no_ace_draw)
        self.Q_ace_hold = np.array(self.Q_ace_hold)
        self.Q_ace_draw = np.array(self.Q_ace_draw)

    def process_data(self, ref):
        self.rmse_no_ace_hold = []
        self.rmse_no_ace_draw = []
        self.rmse_ace_hold = []
        self.rmse_ace_draw = []
        self.rmse_total = []


        for i in range(len(self.t)):

            size = self.Q_no_ace_hold[i].size

            sum1 = np.sum(np.power(ref.Q_no_ace_hold - self.Q_no_ace_hold[i], 2))
            rmse = np.sqrt(sum1 / size)
            self.rmse_no_ace_hold.append(rmse)

            sum2 = np.sum(np.power(ref.Q_no_ace_draw - self.Q_no_ace_draw[i], 2))
            rmse = np.sqrt(sum2 / size)
            self.rmse_no_ace_draw.append(rmse)

            sum3 = np.sum(np.power(ref.Q_ace_hold - self.Q_ace_hold[i], 2))
            rmse = np.sqrt(sum3 / size)
            self.rmse_ace_hold.append(rmse)

            sum4 = np.sum(np.power(ref.Q_ace_draw - self.Q_ace_draw[i], 2))
            rmse = np.sqrt(sum4 / size)
            self.rmse_ace_draw.append(rmse)

            rmse_total = np.sqrt( (sum1 + sum2 + sum3 + sum4) / size * 4 )
            self.rmse_total.append(rmse_total)

    def save(self, filename):
        self.prep_to_save()

        with open(filename, 'wb') as f:
            pickle.dump( self.__dict__, f )

    def load(self, filename):
        with open(filename, 'rb') as f:
            tmp_dict = pickle.load(f)
            self.__dict__.clear()
            self.__dict__.update(tmp_dict)

