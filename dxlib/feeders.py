import numpy

'''Code originally downloaded, and modified from:
from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
Originally distributed under Apache Licence 2.0
'''

class Feeder:
    '''Breaks large dataset into small batches.

    Will auto-shuffle and loop-around.

    Example:
        features = np.array([[ 0,  1,  2],
                             [10, 11, 12],
                             [20, 21, 22],
                             [30, 31, 32],
                             [40, 41, 42],
                             [50, 51, 52],
                             [60, 61, 62],
                             [70, 71, 72],
                             [80, 81, 82],
                             [90, 91, 92]], dtype=np.float32)
        labels = np.array( [[ 0],
                            [ 1],
                            [ 2],
                            [ 3],
                            [ 4],
                            [ 5],
                            [ 6],
                            [ 7],
                            [ 8],
                            [ 9]], dtype=np.int32)
        fd = feeders.Feeder(self.features, self.labels)
        feature_batch, label_batch = fd.next_batch(3)
    '''
    def __init__(self, features, labels):
        if len(features) != len(labels):
            raise ValueError('Featurs/labels lengths must be equal.')

        self._features = features
        self._labels = labels

        self._num_examples = len(features)
        self._epochs_completed = 0
        self._batches_completed = 0
        self._index_in_epoch = 0

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def batches_completed(self):
        return self._batches_completed


    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        self._batches_completed += 1

        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._features = self.features[perm0]
            self._labels = self.labels[perm0]

        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1

            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._features[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]

            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._features = self.features[perm]
                self._labels = self.labels[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._features[start:end]
            labels_new_part = self._labels[start:end]
            return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
        
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch

        return self._features[start:end], self._labels[start:end]