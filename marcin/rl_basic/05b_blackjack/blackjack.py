import numpy as np
import matplotlib.pyplot as plt
import pdb

class Card:
    ACE = 'Ace'; 
    N_2 = '2'; N_3 = '3'; N_4 = '4'; N_5 = '5';
    N_6 = '6'; N_7 = '7'; N_8 = '8'; N_9 = '9'; N_10 = '10';
    JACK = 'Jack'; QUEEN = 'Queen'; KING = 'King'
    _names = \
        [ACE, N_2, N_3, N_4, N_5, N_6, N_7, N_8, N_9, N_10, JACK, QUEEN, KING]
    _values = \
        [  1,   2  , 3,   4,   5,   6,   7,   8,   9,   10,   10,    10,   10]

    def __init__(self, rnd=None):
        if rnd is None:
            rnd = np.random.randint(0, len(Card._names))
        else:
            rnd = Card._names.index(rnd)

        self.name = Card._names[rnd]
        self.value = Card._values[rnd]

        if rnd == 0:
            self.is_ace = True
        else:
            self.is_ace = False

    def __str__(self):
        return self.name

class Hand:
    def __init__(self):
        self.cards = []
        self.points = 0
        self.has_usabe_ace = 0

    def _update(self):
        num_aces = 0
        self.has_usabe_ace = 0
        self.points = 0

        for c in self.cards:
            if c.is_ace:
                num_aces += 1
            self.points += c.value

        if num_aces > 0 and self.points <= 11:
            self.points += 10
            self.has_usabe_ace = 1

    def draw(self, rnd=None):
        self.cards.append(Card(rnd))
        self._update()

    def __str__(self):
        res = ''
        for c in self.cards:
            res += str(c) + ' '

        res += '(' + str(self.points) + ' ' + str(self.has_usabe_ace) + ')'

        return res

class BlackjackEnv:
    """Blackjack game environment. API close to OpenAI Gym."""

    def __init__(self):
        # self._player_hidden_sequence = []  # for testing only
        # self._dealer_hidden_sequence = []  # for testing only
        self.player_hand = None
        self.dealer_hand = None
        self.finished = True
        self.t_step = 0

    def _game_step(self, action):
        self.t_step += 1
        if action == 1:  # draw
            self.player_hand.draw()

            if self.player_hand.points > 21:
                # player busted
                return -1, True
            else:
                # continue game
                return 0, False

        else:  # action == False
            # player done, dealer plays till end
            while self.dealer_hand.points <= 16:
                self.dealer_hand.draw()

            if self.dealer_hand.points > 21:
                return 1, True
            elif self.dealer_hand.points > self.player_hand.points:
                return -1, True  # draw
            elif self.dealer_hand.points == self.player_hand.points:
                return 0, True
            else:
                return 1, True

        raise ValueError('this should not happen')

    def _print_debug(self):
        print('Env print, t=' + str(self.t_step) + ':')
        print('Player:', self.player_hand)
        print('Dealer:', self.dealer_hand)
        print('Finished', self.finished)

    def reset(self):
        """Reset environment

        Returns:
            player points (int, 12-21): player points
        """
        self.finished = False

        self.dealer_hand = Hand()
        self.dealer_hand.draw()
        self.dealer_hand.draw()

        self.player_hand = Hand()
        while self.player_hand.points < 12:
            # Player picks cards until at least 12 points
            self.player_hand.draw()

        self.t_step = 0

        obs = ( self.player_hand.has_usabe_ace,
                self.player_hand.points,
                self.dealer_hand.cards[0].value )
        return obs

    def reset_exploring_starts(self):
        has_ace = np.random.randint(0, 2)
        player_points = np.random.randint(12, 22)
        dealer_shows = np.random.randint(1, 11)

        self.finished = False

        self.dealer_hand = Hand()
        self.dealer_hand.draw(Card._names[dealer_shows-1])
        self.dealer_hand.draw()

        self.player_hand = Hand()
        if not has_ace:
            if player_points == 12:
                self.player_hand.draw(Card.N_10)
                self.player_hand.draw(Card.N_2)
            elif player_points == 13:
                self.player_hand.draw(Card.N_10)
                self.player_hand.draw(Card.N_3)
            elif player_points == 14:
                self.player_hand.draw(Card.N_10)
                self.player_hand.draw(Card.N_4)
            elif player_points == 15:
                self.player_hand.draw(Card.N_10)
                self.player_hand.draw(Card.N_5)
            elif player_points == 16:
                self.player_hand.draw(Card.N_10)
                self.player_hand.draw(Card.N_6)
            elif player_points == 17:
                self.player_hand.draw(Card.N_10)
                self.player_hand.draw(Card.N_7)
            elif player_points == 18:
                self.player_hand.draw(Card.N_10)
                self.player_hand.draw(Card.N_8)
            elif player_points == 19:
                self.player_hand.draw(Card.N_10)
                self.player_hand.draw(Card.N_9)
            elif player_points == 20:
                self.player_hand.draw(Card.N_10)
                self.player_hand.draw(Card.N_10)
            elif player_points == 21:
                self.player_hand.draw(Card.N_10)
                self.player_hand.draw(Card.N_9)
                self.player_hand.draw(Card.N_2)
        else:
            # has_ace == 1
            if player_points == 12:
                self.player_hand.draw(Card.ACE)
                self.player_hand.draw(Card.ACE)
            elif player_points == 13:
                self.player_hand.draw(Card.N_2)
                self.player_hand.draw(Card.ACE)
            elif player_points == 14:
                self.player_hand.draw(Card.N_3)
                self.player_hand.draw(Card.ACE)
            elif player_points == 15:
                self.player_hand.draw(Card.N_4)
                self.player_hand.draw(Card.ACE)
            elif player_points == 16:
                self.player_hand.draw(Card.N_5)
                self.player_hand.draw(Card.ACE)
            elif player_points == 17:
                self.player_hand.draw(Card.N_6)
                self.player_hand.draw(Card.ACE)
            elif player_points == 18:
                self.player_hand.draw(Card.N_7)
                self.player_hand.draw(Card.ACE)
            elif player_points == 19:
                self.player_hand.draw(Card.N_8)
                self.player_hand.draw(Card.ACE)
            elif player_points == 20:
                self.player_hand.draw(Card.N_9)
                self.player_hand.draw(Card.ACE)
            elif player_points == 21:
                self.player_hand.draw(Card.N_10)
                self.player_hand.draw(Card.ACE)


        self.t_step = 0

        if not self.player_hand.has_usabe_ace == has_ace:
            pdb.set_trace()

        assert self.player_hand.has_usabe_ace == has_ace
        assert self.player_hand.points == player_points
        assert self.dealer_hand.cards[0].value == dealer_shows

        obs = ( self.player_hand.has_usabe_ace,
                self.player_hand.points,
                self.dealer_hand.cards[0].value )
        return obs

    def step(self, action):
        """Take action and roll environment one t-step

        Params:
            actions (Action): Action object

        Returns:
            obs (list of float): player points, player pace, dealer points
            reward (float): reward for action 
            done (bool): True if episode completed, False otherwise
            info: always None, for compatibility with OpenAI Gym
        """
        if self.finished:
            raise ValueError('Trying to play game step, but game ended.')
        
        reward, self.finished = self._game_step(action)
        obs = ( self.player_hand.has_usabe_ace,
                self.player_hand.points,
                self.dealer_hand.cards[0].value)

        return obs, reward, self.finished
