import unittest
import feeders

import numpy as np

class FeedersTest(unittest.TestCase):

    def setUp(self):
        self.features = np.array([[ 0,  1,  2],
                                  [10, 11, 12],
                                  [20, 21, 22],
                                  [30, 31, 32],
                                  [40, 41, 42],
                                  [50, 51, 52],
                                  [60, 61, 62],
                                  [70, 71, 72],
                                  [80, 81, 82],
                                  [90, 91, 92]], dtype=np.float32)

        self.labels = np.array([[ 0],
                                [ 1],
                                [ 2],
                                [ 3],
                                [ 4],
                                [ 5],
                                [ 6],
                                [ 7],
                                [ 8],
                                [ 9]], dtype=np.int32)

    def test_construtor(self):
        # labels wrong length
        bad_labels =  np.array([[ 0],
                                [ 1],
                                [ 2],
                                [ 3]], dtype=np.int32)
        self.assertRaises(ValueError, feeders.Feeder,
            self.features, bad_labels)

    def test_next_batch_no_shuffle(self):
        fd = feeders.Feeder(self.features, self.labels)

        #
        #   Begin dataset
        #
        features, labels = fd.next_batch(3, shuffle=False)
        features_ok = np.array([[ 0,  1,  2],
                                [10, 11, 12],
                                [20, 21, 22]], dtype=np.float32)
        labels_ok = np.array([[ 0], [ 1], [ 2]], dtype=np.int32)

        self.assertEqual(len(features), 3)
        self.assertEqual(len(labels), 3)
        self.assertTrue( (features==features_ok).all() )
        self.assertTrue( (labels==labels_ok).all() )


        #
        #   Loop-around case
        #
        for i in range(2):
            f, l = fd.next_batch(3, shuffle=False)

        features, labels = fd.next_batch(3, shuffle=False)

        features_ok = np.array([[90, 91, 92],
                                [ 0,  1,  2],
                                [10, 11, 12]], dtype=np.float32)
        labels_ok = np.array([[ 9], [ 0], [ 1]], dtype=np.int32)

        self.assertEqual(len(features), 3)
        self.assertEqual(len(labels), 3)
        self.assertTrue( (features==features_ok).all() )
        self.assertTrue( (labels==labels_ok).all() )

    def test_next_batch_with_shuffle(self):
        np.random.seed(0)

        fd = feeders.Feeder(self.features, self.labels)

        all_labels = []

        #
        #   Begin dataset
        #
        features, labels = fd.next_batch(3)
        features_ok = np.array([[20, 21, 22],
                                [80, 81, 82],
                                [40, 41, 42]], dtype=np.float32)
        labels_ok = np.array([[ 2], [ 8], [ 4]], dtype=np.int32)

        self.assertEqual(len(features), 3)
        self.assertEqual(len(labels), 3)
        self.assertTrue( (features==features_ok).all() )
        self.assertTrue( (labels==labels_ok).all() )

        all_labels.append(labels)


        #
        #   Loop a bit, feed total of 30 labels, including first feed above
        #
        for i in range(9):
            f, l = fd.next_batch(3, shuffle=False)
            all_labels.append(l)

        all_labels = np.array(all_labels)
        all_labels = all_labels.flatten()

        all_unique = np.unique(all_labels)
        all_count  = np.bincount(all_labels)

        unique_ok = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        count_ok = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3])

        self.assertEqual(len(all_labels), 30)
        self.assertEqual(len(all_unique), 10)
        self.assertEqual(len(all_count), 10)

        self.assertTrue( (all_unique==unique_ok).all() )
        self.assertTrue( (all_count==count_ok).all() )    

if __name__ == '__main__':
    unittest.main()