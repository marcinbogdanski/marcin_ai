import unittest
from blackjack import Card, Hand, BlackjackEnv

class BlackjackTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_card(self):
        self.assertEqual(Card(Card.ACE).value, 1)
        self.assertEqual(Card(Card.ACE).is_ace, True)
        with self.assertRaises(ValueError):
            Card(rnd=13)

    def test_hand(self):
        hand = Hand()
        hand.draw(Card.N_10)
        self.assertEqual(hand.points, 10)
        self.assertEqual(hand.has_usabe_ace, False)
        hand.draw(Card.JACK)
        self.assertEqual(hand.points, 20)
        self.assertEqual(hand.has_usabe_ace, False)
        hand.draw(Card.N_2)
        self.assertEqual(hand.points, 22)
        self.assertEqual(hand.has_usabe_ace, False)

        hand = Hand()
        hand.draw(Card.QUEEN)
        hand.draw(Card.KING)
        hand.draw(Card.ACE)
        self.assertEqual(hand.points, 21)
        self.assertEqual(hand.has_usabe_ace, False)

        hand = Hand()
        hand.draw(Card.N_2)
        hand.draw(Card.ACE)
        self.assertEqual(hand.points, 13)
        self.assertEqual(hand.has_usabe_ace, True)

        hand = Hand()
        hand.draw(Card.N_10)
        hand.draw(Card.N_2)
        hand.draw(Card.ACE)
        self.assertEqual(hand.points, 13)
        self.assertEqual(hand.has_usabe_ace, False)

if __name__ == '__main__':
    unittest.main()