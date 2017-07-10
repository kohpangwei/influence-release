import numpy as np
import os
import time

import IPython
from scipy.stats import pearsonr


def get_try_check(model, X_train, Y_train, Y_train_flipped, X_test, Y_test):
    def try_check(idx_to_check, label):
        Y_train_fixed = np.copy(Y_train_flipped)
        Y_train_fixed[idx_to_check] = Y_train[idx_to_check]
        model.update_train_x_y(X_train, Y_train_fixed)
        model.train()
        check_num = np.sum(Y_train_fixed != Y_train_flipped)
        check_loss, check_acc = model.sess.run(
            [model.loss_no_reg, model.accuracy_op], 
            feed_dict=model.all_test_feed_dict)

        print('%20s: fixed %3s labels. Loss %.5f. Accuracy %.3f.' % (
            label, check_num, check_loss, check_acc))
        return check_num, check_loss, check_acc
    return try_check


def test_mislabeled_detection_batch(
    model, 
    X_train, Y_train,
    Y_train_flipped,
    X_test, Y_test, 
    train_losses, train_loo_influences,
    num_flips, num_checks):    

    assert num_checks > 0

    num_train_examples = Y_train.shape[0] 
    
    try_check = get_try_check(model, X_train, Y_train, Y_train_flipped, X_test, Y_test)

    # Pick by LOO influence    
    idx_to_check = np.argsort(train_loo_influences)[-num_checks:]
    fixed_influence_loo_results = try_check(idx_to_check, 'Influence (LOO)')

    # Pick by top loss to fix
    idx_to_check = np.argsort(np.abs(train_losses))[-num_checks:]    
    fixed_loss_results = try_check(idx_to_check, 'Loss')

    # Randomly pick stuff to fix
    idx_to_check = np.random.choice(num_train_examples, size=num_checks, replace=False)    
    fixed_random_results = try_check(idx_to_check, 'Random')
    
    return fixed_influence_loo_results, fixed_loss_results, fixed_random_results




def viz_top_influential_examples(model, test_idx):

    model.reset_datasets()
    print('Test point %s has label %s.' % (test_idx, model.data_sets.test.labels[test_idx]))

    num_to_remove = 10000
    indices_to_remove = np.arange(num_to_remove)
    
    predicted_loss_diffs = model.get_influence_on_test_loss(
        test_idx, 
        indices_to_remove,
        force_refresh=True)

    # If the predicted difference in loss is high (very positive) after removal,
    # that means that the point helped it to be correct.
    top_k = 10
    helpful_points = np.argsort(predicted_loss_diffs)[-top_k:][::-1]
    unhelpful_points = np.argsort(predicted_loss_diffs)[:top_k]

    for points, message in [
        (helpful_points, 'better'), (unhelpful_points, 'worse')]:
        print("Top %s training points making the loss on the test point %s:" % (top_k, message))
        for counter, idx in enumerate(points):
            print("#%s, class=%s, predicted_loss_diff=%.8f" % (
                idx, 
                model.data_sets.train.labels[idx], 
                predicted_loss_diffs[idx]))




def test_retraining(model, test_idx, iter_to_load, force_refresh=False, 
                    num_to_remove=50, num_steps=1000, random_seed=17,
                    remove_type='random'):

    np.random.seed(random_seed)

    model.load_checkpoint(iter_to_load)
    sess = model.sess

    
    y_test = model.data_sets.test.labels[test_idx]
    print('Test label: %s' % y_test)

    ## Or, randomly remove training examples
    if remove_type == 'random':
        indices_to_remove = np.random.choice(model.num_train_examples, size=num_to_remove, replace=False)
        predicted_loss_diffs = model.get_influence_on_test_loss(
            [test_idx], 
            indices_to_remove,
            force_refresh=force_refresh)
    ## Or, remove the most influential training examples
    elif remove_type == 'maxinf':    
        predicted_loss_diffs = model.get_influence_on_test_loss(
            [test_idx], 
            np.arange(len(model.data_sets.train.labels)),
            force_refresh=force_refresh)
        indices_to_remove = np.argsort(np.abs(predicted_loss_diffs))[-num_to_remove:]
        predicted_loss_diffs = predicted_loss_diffs[indices_to_remove]
    else:
        raise ValueError, 'remove_type not well specified'
    actual_loss_diffs = np.zeros([num_to_remove])

    # Sanity check
    test_feed_dict = model.fill_feed_dict_with_one_ex(
        model.data_sets.test,  
        test_idx)    
    test_loss_val, params_val = sess.run([model.loss_no_reg, model.params], feed_dict=test_feed_dict)
    train_loss_val = sess.run(model.total_loss, feed_dict=model.all_train_feed_dict)
    # train_loss_val = model.minibatch_mean_eval([model.total_loss], model.data_sets.train)[0]

    model.retrain(num_steps=num_steps, feed_dict=model.all_train_feed_dict)
    retrained_test_loss_val = sess.run(model.loss_no_reg, feed_dict=test_feed_dict)
    retrained_train_loss_val = sess.run(model.total_loss, feed_dict=model.all_train_feed_dict)
    # retrained_train_loss_val = model.minibatch_mean_eval([model.total_loss], model.data_sets.train)[0]

    model.load_checkpoint(iter_to_load, do_checks=False)

    print('Sanity check: what happens if you train the model a bit more?')
    print('Loss on test idx with original model    : %s' % test_loss_val)
    print('Loss on test idx with retrained model   : %s' % retrained_test_loss_val)
    print('Difference in test loss after retraining     : %s' % (retrained_test_loss_val - test_loss_val))
    print('===')
    print('Total loss on training set with original model    : %s' % train_loss_val)
    print('Total loss on training with retrained model   : %s' % retrained_train_loss_val)
    print('Difference in train loss after retraining     : %s' % (retrained_train_loss_val - train_loss_val))
    
    print('These differences should be close to 0.\n')

    # Retraining experiment
    for counter, idx_to_remove in enumerate(indices_to_remove):

        print("=== #%s ===" % counter)
        print('Retraining without train_idx %s (label %s):' % (idx_to_remove, model.data_sets.train.labels[idx_to_remove]))

        train_feed_dict = model.fill_feed_dict_with_all_but_one_ex(model.data_sets.train, idx_to_remove)
        model.retrain(num_steps=num_steps, feed_dict=train_feed_dict)
        retrained_test_loss_val, retrained_params_val = sess.run([model.loss_no_reg, model.params], feed_dict=test_feed_dict)
        actual_loss_diffs[counter] = retrained_test_loss_val - test_loss_val

        print('Diff in params: %s' % np.linalg.norm(np.concatenate(params_val) - np.concatenate(retrained_params_val)))      
        print('Loss on test idx with original model    : %s' % test_loss_val)
        print('Loss on test idx with retrained model   : %s' % retrained_test_loss_val)
        print('Difference in loss after retraining     : %s' % actual_loss_diffs[counter])
        print('Predicted difference in loss (influence): %s' % predicted_loss_diffs[counter])

        # Restore params
        model.load_checkpoint(iter_to_load, do_checks=False)
        

    np.savez(
        'output/%s_loss_diffs' % model.model_name, 
        actual_loss_diffs=actual_loss_diffs, 
        predicted_loss_diffs=predicted_loss_diffs)

    print('Correlation is %s' % pearsonr(actual_loss_diffs, predicted_loss_diffs)[0])
    return actual_loss_diffs, predicted_loss_diffs, indices_to_remove