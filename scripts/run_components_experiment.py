from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import copy
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base

from load_mnist import load_mnist

from influence.binaryLogisticRegressionWithLBFGS import BinaryLogisticRegressionWithLBFGS
import influence.dataset as dataset
from influence.dataset import DataSet
from influence.image_utils import plot_flat_bwimage, plot_flat_colorimage, plot_flat_colorgrad



data_sets = load_mnist('data')

pos_class = 1
neg_class = 7

X_train = data_sets.train.x
Y_train = data_sets.train.labels
X_test = data_sets.test.x
Y_test = data_sets.test.labels

X_train, Y_train = dataset.filter_dataset(X_train, Y_train, pos_class, neg_class)
X_test, Y_test = dataset.filter_dataset(X_test, Y_test, pos_class, neg_class)

lr_train = DataSet(X_train, np.array((Y_train + 1) / 2, dtype=int))
lr_validation = None
lr_test = DataSet(X_test, np.array((Y_test + 1) / 2, dtype=int))
lr_data_sets = base.Datasets(train=lr_train, validation=lr_validation, test=lr_test)

num_classes = 2
input_side = 28
input_channels = 1
input_dim = input_side * input_side * input_channels
weight_decay = 0.01
batch_size = 100
initial_learning_rate = 0.001 
keep_probs = None
decay_epochs = [1000, 10000]
max_lbfgs_iter = 1000

num_params = 784

tf.reset_default_graph()

tf_model = BinaryLogisticRegressionWithLBFGS(
    input_dim=input_dim,
    weight_decay=weight_decay,
    max_lbfgs_iter=max_lbfgs_iter,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=lr_data_sets,
    initial_learning_rate=initial_learning_rate,
    keep_probs=keep_probs,
    decay_epochs=decay_epochs,
    mini_batch=False,
    train_dir='output',
    log_dir='log',
    model_name='mnist-17_logreg')

tf_model.train()


test_idx = 30
num_train = len(tf_model.data_sets.train.labels)

influences = tf_model.get_influence_on_test_loss(
    [test_idx], 
    np.arange(len(tf_model.data_sets.train.labels)),
    force_refresh=True) * num_train

influences_without_train_error = tf_model.get_influence_on_test_loss(
    [test_idx], 
    np.arange(len(tf_model.data_sets.train.labels)),
    force_refresh=False,
    ignore_training_error=True) * num_train

influences_without_hessian = tf_model.get_influence_on_test_loss(
    [test_idx], 
    np.arange(len(tf_model.data_sets.train.labels)),
    force_refresh=False,
    ignore_hessian=True) * num_train

influences_without_both = tf_model.get_influence_on_test_loss(
    [test_idx], 
    np.arange(len(tf_model.data_sets.train.labels)),
    force_refresh=False,
    ignore_training_error=True,
    ignore_hessian=True) * num_train

x_test = tf_model.data_sets.test.x[test_idx, :]
y_test = tf_model.data_sets.test.labels[test_idx]
flipped_idx = tf_model.data_sets.train.labels != y_test


test_image = tf_model.data_sets.test.x[test_idx, :]

# Find a training image that has the same label but is harmful
train_idx = np.where(~flipped_idx)[0][np.argsort(influences[~flipped_idx])[4]]
harmful_train_image = tf_model.data_sets.train.x[train_idx, :]

np.savez('output/components_results',
    influences=influences,
    influences_without_train_error=influences_without_train_error,
    influences_without_hessian=influences_without_hessian,
    influences_without_both=influences_without_both,
    flipped_idx=flipped_idx,
    test_image=test_image,
    harmful_train_image=harmful_train_image)