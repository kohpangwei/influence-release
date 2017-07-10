from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  
   
import numpy as np
import pandas as pd
import tensorflow as tf
 
from scipy.stats import pearsonr
from load_mnist import load_mnist

import influence.experiments as experiments
import influence.dataset as dataset
from influence.dataset import DataSet
from influence.smooth_hinge import SmoothHinge
from influence.binaryLogisticRegressionWithLBFGS import BinaryLogisticRegressionWithLBFGS

from tensorflow.contrib.learn.python.learn.datasets import base


data_sets = load_mnist('data')

pos_class = 1
neg_class = 7

X_train = data_sets.train.x
Y_train = data_sets.train.labels
X_test = data_sets.test.x
Y_test = data_sets.test.labels

X_train, Y_train = dataset.filter_dataset(X_train, Y_train, pos_class, neg_class)
X_test, Y_test = dataset.filter_dataset(X_test, Y_test, pos_class, neg_class)

# Round dataset size off to the nearest 100, just for batching convenience 
num_train = int(np.floor(len(Y_train) / 100) * 100)
num_test = int(np.floor(len(Y_test) / 100) * 100)
X_train = X_train[:num_train, :]
Y_train = Y_train[:num_train]
X_test = X_test[:num_test, :]
Y_test = Y_test[:num_test]

train = DataSet(X_train, Y_train)
validation = None
test = DataSet(X_test, Y_test)
data_sets = base.Datasets(train=train, validation=validation, test=test)


num_classes = 2
input_side = 28
input_channels = 1
input_dim = input_side * input_side * input_channels
weight_decay = 0.01
use_bias = False
batch_size = 100
initial_learning_rate = 0.001 
keep_probs = None
decay_epochs = [1000, 10000]
max_lbfgs_iter = 1000

temps = [0, 0.001, 0.1]
num_temps = len(temps)

num_params = 784


# Get weights from hinge

tf.reset_default_graph()

temp = 0
model = SmoothHinge(
    use_bias=use_bias,
    temp=temp,
    input_dim=input_dim,
    weight_decay=weight_decay,
    num_classes=num_classes,
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    keep_probs=keep_probs,
    decay_epochs=decay_epochs,
    mini_batch=False,
    train_dir='output',
    log_dir='log',
    model_name='smooth_hinge_17_t-%s' % temp)

model.train()
model.load_checkpoint(iter_to_load=0)
hinge_W = model.sess.run(model.params)[0]

model_margins = model.sess.run(model.margin, feed_dict=model.all_test_feed_dict)
# Look at np.argsort(model_margins)[:10] to pick a test example

np.random.seed(92)
num_to_remove = 100

params = np.zeros([num_temps, num_params])
margins = np.zeros([num_temps, num_train])
influences = np.zeros([num_temps, num_train])

actual_loss_diffs = np.zeros([num_temps, num_to_remove])
predicted_loss_diffs = np.zeros([num_temps, num_to_remove])
indices_to_remove = np.zeros([num_temps, num_to_remove], dtype=np.int32)

test_idx = 1597

for counter, temp in enumerate(temps):

    tf.reset_default_graph()

    model = SmoothHinge(
        use_bias=use_bias,
        temp=temp,
        input_dim=input_dim,
        weight_decay=weight_decay,
        num_classes=num_classes,
        batch_size=batch_size,
        data_sets=data_sets,
        initial_learning_rate=initial_learning_rate,
        keep_probs=keep_probs,
        decay_epochs=decay_epochs,
        mini_batch=False,
        train_dir='output',
        log_dir='log',
        model_name='smooth_hinge_17_t-%s' % temp)
  
    if temp == 0:
        model.load_checkpoint(iter_to_load=0)
    else:
        params_feed_dict = {}
        params_feed_dict[model.W_placeholder] = hinge_W
        model.sess.run(model.set_params_op, feed_dict=params_feed_dict)
        model.print_model_eval()
        
    cur_params, cur_margins = model.sess.run([model.params, model.margin], feed_dict=model.all_train_feed_dict)
    cur_influences = model.get_influence_on_test_loss(
        test_indices=[test_idx], 
        train_idx=np.arange(num_train), 
        force_refresh=False)
    
    params[counter, :] = np.concatenate(cur_params)
    margins[counter, :] = cur_margins
    influences[counter, :] = cur_influences
    
    if temp == 0:
        actual_loss_diffs[counter, :], predicted_loss_diffs[counter, :], indices_to_remove[counter, :] = experiments.test_retraining(
            model, 
            test_idx, 
            iter_to_load=0,
            force_refresh=False,
            num_steps=2000,
            remove_type='maxinf',
            num_to_remove=num_to_remove)

np.savez(
    'output/hinge_results', 
    temps=temps,
    indices_to_remove=indices_to_remove,
    actual_loss_diffs=actual_loss_diffs,
    predicted_loss_diffs=predicted_loss_diffs,
    influences=influences
)