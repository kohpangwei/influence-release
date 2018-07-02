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
remove_multiplicity = 100
remove_type = "random"

input_dim = data_sets.train.x.shape[1]
weight_decay = 0.01
batch_size = 1400
initial_learning_rate = 0.001
keep_probs = None
max_lbfgs_iter = 1000
decay_epochs = [1000, 10000]
num_to_remove = 500

if len(sys.argv) > 1:
    remove_multiplicity = int(sys.argv[1])
if len(sys.argv) > 2:
    num_to_remove = int(sys.argv[2])

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
    model_name='mnist_logreg_sensing')

tf_model.train()

test_idx = 8
actual_loss_diffs, predicted_loss_diffs_cg, indices_to_remove, all_loss_diffs = experiments.test_retraining(
    tf_model,
    test_idx,
    iter_to_load=0,
    force_refresh=False,
    num_to_remove=num_to_remove,
    remove_type=remove_type,
    remove_multiplicity=remove_multiplicity,
    random_seed=0)

np.savez(
    'output/mnist_logreg_sensing_{}_{}.npz'.format(remove_multiplicity, num_to_remove),
    actual_loss_diffs=actual_loss_diffs,
    predicted_loss_diffs_cg=predicted_loss_diffs_cg,
    all_loss_diffs=all_loss_diffs,
    indices_to_remove=indices_to_remove,
    )
