import os

from influence.dataset import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base
import numpy as np

from influence.nlprocessor import NLProcessor

from scipy.io import savemat

import IPython

def init_lists(folder):
    a_list = []
    file_list = os.listdir(folder)
    for a_file in file_list:
        f = open(folder + a_file, 'r')
        a_list.append(f.read().decode("latin-1"))
    f.close()
    return a_list


def process_spam(n = None):

    np.random.seed(0)

    nlprocessor = NLProcessor()

    spam = init_lists('data/spam/enron1/spam/')
    ham = init_lists('data/spam/enron1/ham/')

    if n is None:
        docs, Y = nlprocessor.process_spam(spam, ham)
    else:
        docs, Y = nlprocessor.process_spam(spam[:n], ham[:n])
    num_examples = len(Y)

    train_fraction = 0.8
    valid_fraction = 0.0
    num_train_examples = int(train_fraction * num_examples)
    num_valid_examples = int(valid_fraction * num_examples)
    num_test_examples = num_examples - num_train_examples - num_valid_examples

    docs_train = docs[:num_train_examples]
    Y_train = Y[:num_train_examples]

    docs_valid = docs[num_train_examples : num_train_examples+num_valid_examples]
    Y_valid = Y[num_train_examples : num_train_examples+num_valid_examples]

    docs_test = docs[-num_test_examples:]
    Y_test = Y[-num_test_examples:]

    assert(len(docs_train) == len(Y_train))
    assert(len(docs_valid) == len(Y_valid))
    assert(len(docs_test) == len(Y_test))
    assert(len(Y_train) + len(Y_valid) + len(Y_test) == num_examples)
    
    nlprocessor.learn_vocab(docs_train)
    X_train = nlprocessor.get_bag_of_words(docs_train)
    X_valid = nlprocessor.get_bag_of_words(docs_valid)
    X_test = nlprocessor.get_bag_of_words(docs_test)

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


def load_spam(n = None):

    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = process_spam(n)

    # Convert them to dense matrices
    X_train = X_train.toarray()
    X_valid = X_valid.toarray()
    X_test = X_test.toarray()

    train = DataSet(X_train, Y_train)
    validation = DataSet(X_valid, Y_valid)
    test = DataSet(X_test, Y_test)

    return base.Datasets(train=train, validation=validation, test=test)
