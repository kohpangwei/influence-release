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


def sample_random_combinations(n_population, k, size=1):
    combinations = set()
    while len(combinations) < size:
        combination = tuple(np.random.choice(n_population, size=k, replace=False))
        combinations.add(combination)
    return list(combinations)

def test_retraining(model, test_idx, iter_to_load, force_refresh=False, 
                    num_to_remove=50, num_steps=1000, random_seed=17,
                    remove_type='random', remove_multiplicity=1, tail_percentile=10):

    np.random.seed(random_seed)

    model.load_checkpoint(iter_to_load)
    sess = model.sess

    y_test = model.data_sets.test.labels[test_idx]
    print('Test label: %s' % y_test)

    if remove_multiplicity < 1:
        raise ValueError("remove_multiplicity must always be >= 1")
    elif remove_multiplicity > 1:
        if remove_type not in ['random', 'tail_pos', 'tail_neg', 'tail']:
            raise ValueError("Removing multiple points not supported with this remove_type")

    ## Or, randomly remove training examples
    if remove_type == 'random':
        if remove_multiplicity == 1:
            indices_to_remove = np.random.choice(model.num_train_examples, size=num_to_remove, replace=False)
        else:
            indices_to_remove = sample_random_combinations(model.num_train_examples, remove_multiplicity, size=num_to_remove)
        # indices_to_remove is either an array of indices or list of tuples of indices

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
    ## Or, remove randomly from the top percentile of positive-influence training examples
    elif remove_type in ['tail_pos', 'tail_neg', 'tail']:
        # Sanity check
        remove_pool_size = int(model.num_train_examples * tail_percentile / 100)
        print(model.num_train_examples)
        if remove_multiplicity == 1 and remove_pool_size < num_to_remove:
            raise ValueError("Tail is not big enough to have this many trials")
        if remove_multiplicity > 1 and remove_pool_size < remove_multiplicity:
            raise ValueError("Tail is not big enough to remove this many examples")

        predicted_loss_diffs = model.get_influence_on_test_loss(
            [test_idx],
            np.arange(len(model.data_sets.train.labels)),
            force_refresh=force_refresh)
        if remove_type == 'tail_pos':
            sorted_indices = np.argsort(predicted_loss_diffs)
        elif remove_type == 'tail_neg':
            sorted_indices = np.argsort(-predicted_loss_diffs)
        elif remove_type == 'tail':
            sorted_indices = np.argsort(np.abs(predicted_loss_diffs))
        remove_pool = sorted_indices[-remove_pool_size:]

        if remove_multiplicity == 1:
            indices_to_remove = np.random.choice(remove_pool, size=num_to_remove, replace=False)
        else:
            indices_to_remove = sample_random_combinations(remove_pool, remove_multiplicity, size=num_to_remove)

        predicted_loss_diffs = model.get_influence_on_test_loss(
            [test_idx],
            indices_to_remove,
            force_refresh=force_refresh)
    else:
        raise ValueError('remove_type not well specified')
    actual_loss_diffs = np.zeros([num_to_remove])

    # Get the influence of each individual train point
    all_loss_diffs = model.get_influence_on_test_loss(
        [test_idx], 
        np.arange(len(model.data_sets.train.labels)),
        force_refresh=force_refresh)

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
        if type(idx_to_remove) is np.int64:
            print('Retraining without train_idx %s (label %s):' % (idx_to_remove, model.data_sets.train.labels[idx_to_remove]))
        else:
            if len(idx_to_remove) <= 3:
                print('Retraining without train_idxs %s (labels %s):' % (idx_to_remove,
                    [model.data_sets.train.labels[idx] for idx in idx_to_remove]))
            else:
                bit_of_idx = ", ".join(map(str, idx_to_remove[:3]))
                bit_of_labels = ", ".join(map(str, [model.data_sets.train.labels[idx] for idx in idx_to_remove[:3]]))
                print('Retraining without train_idxs (%s, ...) (labels [%s, ...]):' % (bit_of_idx, bit_of_labels))

        train_feed_dict = model.fill_feed_dict_with_all_but_some_ex(model.data_sets.train, idx_to_remove)
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
    return actual_loss_diffs, predicted_loss_diffs, indices_to_remove, all_loss_diffs

def test_retraining_repeated(model, test_idx, repeat_idx=10, repeats=[1, 2, 4],
                             train_subset=None,
                             iter_to_load=0, num_steps=1000, random_seed=17):

    np.random.seed(random_seed)

    model.load_checkpoint(iter_to_load)
    sess = model.sess

    y_test = model.data_sets.test.labels[test_idx]
    print('Test label: %s' % y_test)

    # Sanity check
    test_feed_dict = model.fill_feed_dict_with_one_ex(
        model.data_sets.test,
        test_idx)
    test_loss_val, params_val = sess.run([model.loss_no_reg, model.params], feed_dict=test_feed_dict)
    train_loss_val = sess.run(model.total_loss, feed_dict=model.all_train_feed_dict)
    # train_loss_val = model.minibatch_mean_eval([model.total_loss], model.data_sets.train)[0]

    if train_subset is None:
        train_indices = list(range(model.num_train_examples))
    else:
        train_indices = train_subset
    all_train_feed_dict = model.fill_feed_dict_with_some_ex(model.data_sets.train, train_indices)

    model.retrain(num_steps=num_steps, feed_dict=all_train_feed_dict)

    # Retraining experiment
    actual_loss_diffs = np.zeros((len(repeats)))
    for counter, repeats_of_idx in enumerate(repeats):
        print("=== #{}: {} repeats ===".format(counter, repeats_of_idx))

        indices_to_train = train_indices + [repeat_idx] * repeats_of_idx

        train_feed_dict = model.fill_feed_dict_with_some_ex(model.data_sets.train, indices_to_train)
        model.retrain(num_steps=num_steps, feed_dict=train_feed_dict)
        retrained_test_loss_val, retrained_params_val = sess.run([model.loss_no_reg, model.params], feed_dict=test_feed_dict)
        actual_loss_diffs[counter] = retrained_test_loss_val - test_loss_val

        print('Diff in params: %s' % np.linalg.norm(np.concatenate(params_val) - np.concatenate(retrained_params_val)))      
        print('Loss on test idx with original model    : %s' % test_loss_val)
        print('Loss on test idx with retrained model   : %s' % retrained_test_loss_val)
        print('Difference in loss after retraining     : %s' % actual_loss_diffs[counter])

        # Restore params
        model.load_checkpoint(iter_to_load, do_checks=False)

    return actual_loss_diffs, repeats

def test_influence_expectation(model,
                               test_idx,
                               indices_to_remove,
                               num_steps=1000,
                               retrain_num_steps=50000,
                               iter_to_switch_to_batch=10000000,
                               iter_to_switch_to_sgd=10000000,
                               seed=17):
    sess = model.sess

    y_test = model.data_sets.test.labels[test_idx]
    print('Test label: %s' % y_test)

    print('Initial training...')
    # Train without removing
    np.random.seed(seed)
    model.data_sets.train.reset_batch()
    model.train(num_steps=num_steps,
                iter_to_switch_to_batch=iter_to_switch_to_batch,
                iter_to_switch_to_sgd=iter_to_switch_to_sgd)

    # Initial data
    test_feed_dict = model.fill_feed_dict_with_one_ex(model.data_sets.test, test_idx)
    test_loss_val, params_val = sess.run([model.loss_no_reg, model.params], feed_dict=test_feed_dict)

    model.retrain(num_steps=retrain_num_steps, feed_dict=model.all_train_feed_dict)
    retrained_test_loss_val = sess.run(model.loss_no_reg, feed_dict=test_feed_dict)

    print('Sanity check: what happens if you train the model a bit more?')
    print('Loss on test idx with original model    : %s' % test_loss_val)
    print('Loss on test idx with retrained model   : %s' % retrained_test_loss_val)
    print('Difference in test loss after retraining     : %s' % (retrained_test_loss_val - test_loss_val))

    # Get test loss influence
    print('\nCalculating predicted test loss...')
    predicted_loss_diffs = model.get_influence_on_test_loss(
        [test_idx],
        indices_to_remove,
        force_refresh=True)

    retrained_test_loss = np.zeros(len(indices_to_remove))
    for counter, idx_to_remove in enumerate(indices_to_remove):
        print('\nRetraining (not from scratch) without train example {}...'.format(idx_to_remove))

        train_feed_dict = model.fill_feed_dict_with_all_but_some_ex(model.data_sets.train, [idx_to_remove])
        model.retrain(num_steps=retrain_num_steps, feed_dict=train_feed_dict)
        retrained_test_loss_val = sess.run(model.loss_no_reg, feed_dict=test_feed_dict)

        retrained_test_loss_val, retrained_params_val = sess.run([model.loss_no_reg, model.params], feed_dict=test_feed_dict)
        retrained_test_loss[counter] = retrained_test_loss_val

        print('Predicted loss diff without example {}: {}'.format(
            idx_to_remove, predicted_loss_diffs[counter]))
        print('Actual loss diff without example {}:    {}'.format(
            idx_to_remove, retrained_test_loss[counter] - test_loss_val))

    return retrained_test_loss - test_loss_val, predicted_loss_diffs
