import IPython
import numpy as np

from load_animals import load_animals, load_dogfish_with_koda, load_dogfish_with_orig_and_koda

import os
from shutil import copyfile

from influence.inceptionModel import BinaryInceptionModel
from influence.binaryLogisticRegressionWithLBFGS import BinaryLogisticRegressionWithLBFGS
import influence.experiments
from influence.dataset import DataSet
from influence.dataset_poisoning import iterative_attack, select_examples_to_attack, get_projection_to_box_around_orig_point, generate_inception_features

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import base


img_side = 299
num_channels = 3
 
initial_learning_rate = 0.001 
keep_probs = None
decay_epochs = [1000, 10000]

weight_decay = 0.001

num_classes = 2
max_lbfgs_iter = 1000

batch_size = 30
dataset_name = 'dogfish_koda'
data_sets = load_dogfish_with_koda()    

full_graph = tf.Graph()
top_graph = tf.Graph()

print('*** Full:')
with full_graph.as_default():
    full_model_name = '%s_inception_wd-%s' % (dataset_name, weight_decay)
    full_model = BinaryInceptionModel(
        img_side=img_side,
        num_channels=num_channels,
        weight_decay=weight_decay,
        num_classes=num_classes, 
        batch_size=batch_size,
        data_sets=data_sets,
        initial_learning_rate=initial_learning_rate,
        keep_probs=keep_probs,
        decay_epochs=decay_epochs,
        mini_batch=True,
        train_dir='output',
        log_dir='log',
        model_name=full_model_name)

    for data_set, label in [
        (data_sets.train, 'train'),
        (data_sets.test, 'test')]:

        inception_features_path = 'output/%s_inception_features_new_%s.npz' % (dataset_name, label)
        if not os.path.exists(inception_features_path):

            print('Inception features do not exist. Generating %s...' % label)
            data_set.reset_batch()
            
            num_examples = data_set.num_examples
            assert num_examples % batch_size == 0

            inception_features_val = generate_inception_features(
                full_model, 
                data_set.x, 
                data_set.labels, 
                batch_size=batch_size)
            
            np.savez(
                inception_features_path, 
                inception_features_val=inception_features_val,
                labels=data_set.labels)


train_f = np.load('output/%s_inception_features_new_train.npz' % dataset_name)
train = DataSet(train_f['inception_features_val'], train_f['labels'])
test_f = np.load('output/%s_inception_features_new_test.npz' % dataset_name)
test = DataSet(test_f['inception_features_val'], test_f['labels'])

validation = None

inception_data_sets = base.Datasets(train=train, validation=validation, test=test)

print('*** Top:')
with top_graph.as_default():
    top_model_name = '%s_inception_onlytop_wd-%s' % (dataset_name, weight_decay)
    input_dim = 2048
    top_model = BinaryLogisticRegressionWithLBFGS(
        input_dim=input_dim,
        weight_decay=weight_decay,
        max_lbfgs_iter=max_lbfgs_iter,
        num_classes=num_classes, 
        batch_size=batch_size,
        data_sets=inception_data_sets,
        initial_learning_rate=initial_learning_rate,
        keep_probs=keep_probs,
        decay_epochs=decay_epochs,
        mini_batch=False,
        train_dir='output',
        log_dir='log',
        model_name=top_model_name)
    top_model.train()
    weights = top_model.sess.run(top_model.weights)
    orig_weight_path = 'output/inception_weights_%s.npy' % top_model_name
    np.save(orig_weight_path, weights)


with full_graph.as_default():
    full_model.load_weights_from_disk(orig_weight_path, do_save=False, do_check=True)
    full_model.reset_datasets()



num_train = len(top_model.data_sets.train.labels)
num_test = len(top_model.data_sets.test.labels)
max_num_to_poison = 10
loss_type = 'normal_loss'

### Try attacking all test examples
step_size = 0.005
test_indices = np.arange(num_test)
test_description = 'all_%s' % dataset_name

# Use top model to quickly generate inverse HVP
with top_graph.as_default():
    top_model.get_influence_on_test_loss(
        test_indices, 
        [0],        
        test_description=test_description,
        force_refresh=True)
copyfile(
    'output/%s-cg-normal_loss-test-%s.npz' % (top_model_name, test_description),
    'output/%s-cg-normal_loss-test-%s.npz' % (full_model_name, test_description))

with full_graph.as_default():
    grad_influence_wrt_input_val = full_model.get_grad_of_influence_wrt_input(
        np.arange(num_train), 
        test_indices, 
        force_refresh=False,
        test_description=test_description,
        loss_type='normal_loss')    
    indices_to_poison = select_examples_to_attack(
        full_model, 
        max_num_to_poison, 
        grad_influence_wrt_input_val,
        step_size=step_size)
print('****')
print('Indices to poison: %s' % indices_to_poison)
print('****')

idx_to_poison = indices_to_poison[0]

orig_X_train_subset = np.copy(full_model.data_sets.train.x[idx_to_poison, :])
orig_X_train_inception_features_subset = np.copy(top_model.data_sets.train.x[idx_to_poison, :])

project_fn = get_projection_to_box_around_orig_point(orig_X_train_subset, box_radius_in_pixels=0.5)

iterative_attack(top_model, full_model, top_graph, full_graph, project_fn, test_indices, test_description=test_description, 
    indices_to_poison=[idx_to_poison],
    num_iter=2000,
    step_size=step_size,
    save_iter=500,
    loss_type='normal_loss')