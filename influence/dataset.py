# Adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py

import numpy as np

class DataSet(object):

    def __init__(self, x, labels):

        if len(x.shape) > 2:
            x = np.reshape(x, [x.shape[0], -1])

        assert(x.shape[0] == labels.shape[0])

        x = x.astype(np.float32)

        self._x = x
        self._x_batch = np.copy(x)
        self._labels = labels
        self._labels_batch = np.copy(labels)
        self._num_examples = x.shape[0]
        self._index_in_epoch = 0

    @property
    def x(self):
        return self._x

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def reset_batch(self):
        self._index_in_epoch = 0        
        self._x_batch = np.copy(self._x)
        self._labels_batch = np.copy(self._labels)

    def next_batch(self, batch_size):
        assert batch_size <= self._num_examples

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:

            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._x_batch = self._x_batch[perm, :]
            self._labels_batch = self._labels_batch[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch
        return self._x_batch[start:end], self._labels_batch[start:end]


def filter_dataset(X, Y, pos_class, neg_class):
    """
    Filters out elements of X and Y that aren't one of pos_class or neg_class
    then transforms labels of Y so that +1 = pos_class, -1 = neg_class.
    """
    assert(X.shape[0] == Y.shape[0])
    assert(len(Y.shape) == 1)

    Y = Y.astype(int)
    
    pos_idx = Y == pos_class
    neg_idx = Y == neg_class        
    Y[pos_idx] = 1
    Y[neg_idx] = -1
    idx_to_keep = pos_idx | neg_idx
    X = X[idx_to_keep, ...]
    Y = Y[idx_to_keep]    
    return (X, Y)    


def find_distances(target, X, theta=None):
    assert len(X.shape) == 2, "X must be 2D, but it is currently %s" % len(X.shape)
    target = np.reshape(target, -1)
    assert X.shape[1] == len(target), \
      "X (%s) and target (%s) must have same feature dimension" % (X.shape[1], len(target))
    
    if theta is None:
        return np.linalg.norm(X - target, axis=1)
    else:
        theta = np.reshape(theta, -1)
        
        # Project onto theta
        return np.abs((X - target).dot(theta))    