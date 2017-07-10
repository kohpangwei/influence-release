from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import abc
import sys

import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, cluster
import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse 
from scipy.special import expit 

import os.path
import time
import tensorflow as tf
import math

from influence.hessians import hessians
from influence.genericNeuralNet import GenericNeuralNet, variable, variable_with_weight_decay
from influence.logisticRegressionWithLBFGS import LogisticRegressionWithLBFGS


class BinaryLogisticRegressionWithLBFGS(LogisticRegressionWithLBFGS):

    def __init__(self, **kwargs):

        super(BinaryLogisticRegressionWithLBFGS, self).__init__(**kwargs)

        C = 1.0 / (self.num_train_examples * self.weight_decay)        
        self.sklearn_model = linear_model.LogisticRegression(
            C=C,
            tol=1e-8,
            fit_intercept=False, 
            solver='lbfgs',
            warm_start=True,
            max_iter=1000)

        C_minus_one = 1.0 / ((self.num_train_examples - 1) * self.weight_decay)
        self.sklearn_model_minus_one = linear_model.LogisticRegression(
            C=C_minus_one,
            tol=1e-8,
            fit_intercept=False, 
            solver='lbfgs',
            warm_start=True,
            max_iter=1000)        


    def inference(self, input):                
        with tf.variable_scope('softmax_linear'):
            weights = variable_with_weight_decay(
                'weights', 
                [self.input_dim],
                stddev=1.0 / math.sqrt(float(self.input_dim)),
                wd=self.weight_decay)            

            logits = tf.matmul(input, tf.reshape(weights, [self.input_dim, 1])) # + biases
            zeros = tf.zeros_like(logits)            
            logits_with_zeros = tf.concat([zeros, logits], 1)

        self.weights = weights

        return logits_with_zeros


    def set_params(self):
        self.W_placeholder = tf.placeholder(
            tf.float32,
            shape=[self.input_dim],
            name='W_placeholder')
        set_weights = tf.assign(self.weights, self.W_placeholder, validate_shape=True)
        return [set_weights]


    # Special-purpose function for paper experiments
    # that has flags for ignoring training error or Hessian
    def get_influence_on_test_loss(self, test_indices, train_idx, 
        approx_type='cg', approx_params=None, force_refresh=True, test_description=None,
        loss_type='normal_loss',
        ignore_training_error=False,
        ignore_hessian=False
        ):

        test_grad_loss_no_reg_val = self.get_test_grad_loss_no_reg_val(test_indices, loss_type=loss_type)

        print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))

        start_time = time.time()

        if test_description is None:
            test_description = test_indices

        approx_filename = os.path.join(self.train_dir, '%s-%s-%s-test-%s.npz' % (self.model_name, approx_type, loss_type, test_description))
        if ignore_hessian == False:
            if os.path.exists(approx_filename) and force_refresh == False:
                inverse_hvp = list(np.load(approx_filename)['inverse_hvp'])
                print('Loaded inverse HVP from %s' % approx_filename)
            else:
                inverse_hvp = self.get_inverse_hvp(
                    test_grad_loss_no_reg_val,
                    approx_type,
                    approx_params)
                np.savez(approx_filename, inverse_hvp=inverse_hvp)
                print('Saved inverse HVP to %s' % approx_filename)
        else:
            inverse_hvp = test_grad_loss_no_reg_val

        duration = time.time() - start_time
        print('Inverse HVP took %s sec' % duration)


        start_time = time.time()

        num_to_remove = len(train_idx)
        predicted_loss_diffs = np.zeros([num_to_remove])
        for counter, idx_to_remove in enumerate(train_idx):            
            
            if ignore_training_error == False:
                single_train_feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, idx_to_remove)      
                train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
            else:
                train_grad_loss_val = [-(self.data_sets.train.labels[idx_to_remove] * 2 - 1) * self.data_sets.train.x[idx_to_remove, :]]
            predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp), np.concatenate(train_grad_loss_val)) / self.num_train_examples
            
        duration = time.time() - start_time
        print('Multiplying by %s train examples took %s sec' % (num_to_remove, duration))

        return predicted_loss_diffs

 
    def get_loo_influences(self):

        X_train = self.data_sets.train.x
        Y_train = self.data_sets.train.labels * 2 - 1
        theta = self.sess.run(self.params)[0]

        # Pre-calculate inverse covariance matrix
        n = X_train.shape[0]        
        dim = X_train.shape[1]
        cov = np.zeros([dim, dim])

        probs = expit(np.dot(X_train, theta.T))
        weighted_X_train = np.reshape(probs * (1 - probs), (-1, 1)) * X_train

        cov = np.dot(X_train.T, weighted_X_train) / n 
        cov += self.weight_decay * np.eye(dim)                    

        cov_lu_factor = slin.lu_factor(cov)

        assert(len(Y_train.shape) == 1)
        x_train_theta = np.reshape(X_train.dot(theta.T), [-1])
        sigma = expit(-Y_train * x_train_theta)

        d_theta = slin.lu_solve(cov_lu_factor, X_train.T).T 

        quad_x = np.sum(X_train * d_theta, axis=1)

        return sigma * quad_x


