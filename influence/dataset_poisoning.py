import IPython
import numpy as np

import os
import time
from shutil import copyfile

from influence.inceptionModel import BinaryInceptionModel
from influence.binaryLogisticRegressionWithLBFGS import BinaryLogisticRegressionWithLBFGS
import influence.experiments
from influence.dataset import DataSet

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import base


def get_projection_to_box_around_orig_point(X_orig, box_radius_in_pixels=0.5):
    box_radius_in_float = box_radius_in_pixels * 2.0 / 255.0

    if X_orig is None:
        lower_bound = -1
        upper_bound = 1
    else:
        lower_bound = np.maximum(
            -np.ones_like(X_orig),
            X_orig - box_radius_in_float)
        upper_bound = np.minimum(
            np.ones_like(X_orig),
            X_orig + box_radius_in_float)

    # Automatically enforces -1, 1 as well
    def project_fn(X):
        return np.clip(X, lower_bound, upper_bound)

    return project_fn


def select_examples_to_attack(model, num_to_poison, grad_influence_wrt_input_val, step_size):

    # diffs = model.data_sets.train.x - np.clip(model.data_sets.train.x - step_size * np.sign(grad_influence_wrt_input_val) * 2.0 / 255.0, -1, 1) 
    # pred_diff = np.sum(diffs * grad_influence_wrt_input_val, axis = 1)    
    # This ignores the clipping, but it's faster    
    pred_diff = np.sum(np.abs(grad_influence_wrt_input_val), axis = 1)
    indices_to_poison = np.argsort(pred_diff)[-1:-num_to_poison-1:-1] # First index is the most effective
    return indices_to_poison
    

        

def poison_with_influence_proj_gradient_step(model, indices_to_poison, grad_influence_wrt_input_val_subset, step_size, project_fn):    
    """
    Returns poisoned_X_train, a subset of model.data_sets.train (marked by indices_to_poison)
    that has been modified by a single gradient step.
    """
    poisoned_X_train_subset = project_fn(
        model.data_sets.train.x[indices_to_poison, :] - step_size * np.sign(grad_influence_wrt_input_val_subset) * 2.0 / 255.0)

    print('-- max: %s, mean: %s, min: %s' % (
        np.max(grad_influence_wrt_input_val_subset),
        np.mean(grad_influence_wrt_input_val_subset),
        np.min(grad_influence_wrt_input_val_subset)))

    return poisoned_X_train_subset


def generate_inception_features(model, poisoned_X_train_subset, labels_subset, batch_size=None):
    poisoned_train = DataSet(poisoned_X_train_subset, labels_subset)    
    poisoned_data_sets = base.Datasets(train=poisoned_train, validation=None, test=None)

    if batch_size == None:
        batch_size = len(labels_subset)

    num_examples = poisoned_data_sets.train.num_examples
    assert num_examples % batch_size == 0
    num_iter = int(num_examples / batch_size)

    poisoned_data_sets.train.reset_batch()

    inception_features_val = []
    for i in xrange(num_iter):
        feed_dict = model.fill_feed_dict_with_batch(poisoned_data_sets.train, batch_size=batch_size)
        inception_features_val_temp = model.sess.run(model.inception_features, feed_dict=feed_dict)
        inception_features_val.append(inception_features_val_temp)

    return np.concatenate(inception_features_val)


def iterative_attack(top_model, full_model, top_graph, full_graph, project_fn, test_indices, test_description=None, 
    indices_to_poison=None,
    num_iter=10,
    step_size=1,
    save_iter=1,
    loss_type='normal_loss',
    early_stop=None):     
    # If early_stop is set and it stops early, returns True
    # Otherwise, returns False

    if test_description is None:
        test_description = test_indices

    if early_stop is not None:
        assert len(test_indices) == 1, 'Early stopping only supported for attacks on a single test index.'

    if len(indices_to_poison) == 1:
        train_idx_str = indices_to_poison
    else:
        train_idx_str = len(indices_to_poison)

    top_model_name = top_model.model_name
    full_model_name = full_model.model_name

    print('Test idx: %s' % test_indices)
    print('Indices to poison: %s' % indices_to_poison)

    # Remove everything but the poisoned train indices from the full model, to save time
    full_model.update_train_x_y(
        full_model.data_sets.train.x[indices_to_poison, :],
        full_model.data_sets.train.labels[indices_to_poison])
    eff_indices_to_poison = np.arange(len(indices_to_poison))
    labels_subset = full_model.data_sets.train.labels[eff_indices_to_poison]

    for attack_iter in range(num_iter):
        print('*** Iter: %s' % attack_iter)
        
        print('Calculating grad...')

        # Use top model to quickly generate inverse HVP
        with top_graph.as_default():
            top_model.get_influence_on_test_loss(
                test_indices, 
                [0],        
                force_refresh=True,
                test_description=test_description,
                loss_type=loss_type)

        copyfile(
            'output/%s-cg-%s-test-%s.npz' % (top_model_name, loss_type, test_description),
            'output/%s-cg-%s-test-%s.npz' % (full_model_name, loss_type, test_description))

        # Use full model to get gradient wrt pixels
        with full_graph.as_default():
            grad_influence_wrt_input_val_subset = full_model.get_grad_of_influence_wrt_input(
                eff_indices_to_poison, 
                test_indices, 
                force_refresh=False,
                test_description=test_description,
                loss_type=loss_type)    
            poisoned_X_train_subset = poison_with_influence_proj_gradient_step(
                full_model, 
                eff_indices_to_poison,
                grad_influence_wrt_input_val_subset,
                step_size,
                project_fn)

        # Update training dataset                     
        with full_graph.as_default():
            full_model.update_train_x(poisoned_X_train_subset)

            inception_X_train = top_model.data_sets.train.x
            inception_X_train_subset = generate_inception_features(full_model, poisoned_X_train_subset, labels_subset)
            inception_X_train[indices_to_poison] = inception_X_train_subset

        with top_graph.as_default():
            top_model.update_train_x(inception_X_train)
        
        # Retrain model
        print('Training...')
        with top_graph.as_default():
            top_model.train()
            weights = top_model.sess.run(top_model.weights)
            weight_path = 'output/inception_weights_%s_attack_%s_testidx-%s.npy' % (top_model_name, loss_type, test_description)
            np.save(weight_path, weights)
        with full_graph.as_default():            
            full_model.load_weights_from_disk(weight_path, do_save=False, do_check=False)


        # Print out attack effectiveness if it's not too expensive
        test_pred = None
        if len(test_indices) < 100:
            with full_graph.as_default():
                test_pred = full_model.sess.run(full_model.preds, feed_dict=full_model.fill_feed_dict_with_some_ex(
                    full_model.data_sets.test,
                    test_indices))
                print('Test pred (full): %s' % test_pred)
            with top_graph.as_default():
                test_pred = top_model.sess.run(top_model.preds, feed_dict=top_model.fill_feed_dict_with_some_ex(
                    top_model.data_sets.test,
                    test_indices))
                print('Test pred (top): %s' % test_pred)

            if ((early_stop is not None) and (len(test_indices) == 1)):
                if test_pred[0, int(full_model.data_sets.test.labels[test_indices])] < early_stop:
                    print('Successfully attacked. Saving and breaking...')
                    np.savez('output/%s_attack_%s_testidx-%s_trainidx-%s_stepsize-%s_proj_final' % (full_model.model_name, loss_type, test_description, train_idx_str, step_size), 
                        poisoned_X_train_image=poisoned_X_train_subset, 
                        poisoned_X_train_inception_features=inception_X_train_subset,
                        Y_train=labels_subset,
                        indices_to_poison=indices_to_poison,
                        attack_iter=attack_iter + 1,
                        test_pred=test_pred,
                        step_size=step_size)            
                    return True

        if (attack_iter+1) % save_iter == 0:
            np.savez('output/%s_attack_%s_testidx-%s_trainidx-%s_stepsize-%s_proj_iter-%s' % (full_model.model_name, loss_type, test_description, train_idx_str, step_size, attack_iter+1), 
                poisoned_X_train_image=poisoned_X_train_subset, 
                poisoned_X_train_inception_features=inception_X_train_subset,
                Y_train=labels_subset,
                indices_to_poison=indices_to_poison,
                attack_iter=attack_iter + 1,
                test_pred=test_pred,
                step_size=step_size)

    return False
