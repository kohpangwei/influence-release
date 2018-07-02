from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import numpy as np
import pandas as pd

import IPython
import tensorflow as tf

from influence.logisticRegressionWithLBFGS import LogisticRegressionWithLBFGS
import influence.experiments as experiments

from load_mnist import load_mnist, load_small_mnist

import sys

data_sets = load_small_mnist('data')

num_classes = 10
remove_multiplicity = 2

input_dim = data_sets.train.x.shape[1]
weight_decay = 0.01
batch_size = 1400
initial_learning_rate = 0.001
keep_probs = None
max_lbfgs_iter = 1000
decay_epochs = [1000, 10000]

if len(sys.argv) > 1:
    remove_multiplicity = int(sys.argv[1])
if len(sys.argv) > 2:
    remove_type = sys.argv[2]

tf.reset_default_graph()

tf_model = LogisticRegressionWithLBFGS(
    input_dim=input_dim,
    weight_decay=weight_decay,
    max_lbfgs_iter=max_lbfgs_iter,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    keep_probs=keep_probs,
    decay_epochs=decay_epochs,
    mini_batch=False,
    train_dir='output',
    log_dir='log',
    model_name='mnist_logreg_lbfgs_repeated')

tf_model.train()

test_idx = 8
repeat_indices = [6, 7, 8, 9, 10, 20, 30, 40, 50]
for repeat_idx in repeat_indices:
    repeats = list(range(20)) + [30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000]
    actual_loss_diffs, repeats = experiments.test_retraining_repeated(
        tf_model,
        test_idx,
        repeat_idx,
        repeats)

    np.savez(
        'output/mnist_logreg_lbfgs_repeated_point-{}.npz'.format(repeat_idx),
        actual_loss_diffs=actual_loss_diffs,
        repeats=repeats,
        )

test_idx = 8
repeat_indices = list(range(40, 101, 10)) + list(range(91, 100))
train_subset = [8, 0, 6, 1, 38, 17, 2, 11, 4, 3] # Need to have 1 of each class
for repeat_idx in repeat_indices:
    repeats = list(range(1, 20)) + [30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000]
    actual_loss_diffs, repeats = experiments.test_retraining_repeated(
        tf_model,
        test_idx,
        repeat_idx,
        repeats,
        train_subset=train_subset + [repeat_idx])

    np.savez(
        'output/mnist_logreg_lbfgs_repeated_point-fight-{}.npz'.format(repeat_idx),
        actual_loss_diffs=actual_loss_diffs,
        repeats=repeats,
        )

