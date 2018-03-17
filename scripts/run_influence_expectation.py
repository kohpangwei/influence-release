from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import numpy as np
import IPython

import tensorflow as tf

import influence.experiments as experiments
from influence.all_CNN_c import All_CNN_C

from load_mnist import load_small_mnist, load_mnist
from tensorflow.contrib.learn.python.learn.datasets import base

data_sets = load_small_mnist('data')

# Make the dataset smaller by a factor of 10
num_train = data_sets.train.num_examples // 10
num_test = data_sets.test.num_examples // 10
num_validation = data_sets.validation.num_examples // 10
data_sets = base.Datasets(
    train=data_sets.train.take(num_train),
    test=data_sets.test.take(num_test),
    validation=data_sets.train.take(num_validation),
)

num_classes = 10
input_side = 28
input_channels = 1
input_dim = input_side * input_side * input_channels 
weight_decay = 0.001
batch_size = 50

initial_learning_rate = 0.0001 
decay_epochs = [10000, 20000]
hidden1_units = 8
hidden2_units = 8
hidden3_units = 8
conv_patch_size = 3
keep_probs = [1.0, 1.0]

model = All_CNN_C(
    input_side=input_side, 
    input_channels=input_channels,
    conv_patch_size=conv_patch_size,
    hidden1_units=hidden1_units, 
    hidden2_units=hidden2_units,
    hidden3_units=hidden3_units,
    weight_decay=weight_decay,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    damping=1e-2,
    decay_epochs=decay_epochs,
    mini_batch=True,
    train_dir='output',
    log_dir='log',
    model_name='mnist_influence_expectation_cnn')

num_steps = 200000
retrain_num_steps = 10000

test_idx = 658
num_to_remove = 5
indices_to_remove = np.random.choice(model.num_train_examples, size=num_to_remove, replace=False)
seeds_to_try = list(range(20))

all_actual_loss_diffs, all_predicted_loss_diffs = [], []
for seed in seeds_to_try:
    actual_loss_diffs, predicted_loss_diffs = experiments.test_influence_expectation(
        model,
        test_idx,
        indices_to_remove,
        num_steps=num_steps,
        retrain_num_steps=retrain_num_steps)

    all_actual_loss_diffs.append(actual_loss_diffs)
    all_predicted_loss_diffs.append(predicted_loss_diffs)

np.savez(
    'output/mnist_influence_expectations_cnn.npz',
    all_actual_loss_diffs=all_actual_loss_diffs,
    all_predicted_loss_diffs=all_predicted_loss_diffs,
    indices_to_remove=indices_to_remove
    )
