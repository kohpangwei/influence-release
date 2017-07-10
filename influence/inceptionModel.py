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

import os.path
import time
import tensorflow as tf
import math

from influence.hessians import hessians
from influence.genericNeuralNet import GenericNeuralNet, variable, variable_with_weight_decay
from influence.inception_v3 import InceptionV3

from keras.layers import Flatten
from keras.layers import AveragePooling2D
from keras.utils.data_utils import get_file
from keras import backend as K


class BinaryInceptionModel(GenericNeuralNet):

    def __init__(self, img_side, num_channels, weight_decay, **kwargs):
        self.weight_decay = weight_decay
        
        self.img_side = img_side
        self.num_channels = num_channels
        self.input_dim = img_side * img_side * num_channels
        self.num_features = 2048 # Hardcoded for inception. For some reason Flatten() doesn't register num_features.

        super(BinaryInceptionModel, self).__init__(**kwargs)

        self.load_inception_weights()
        # Do we need to set trainable to False?
        # We might be unnecessarily blowing up the graph by including all of the train operations
        # needed for the inception network.

        self.set_params_op = self.set_params()

        C = 1.0 / ((self.num_train_examples) * self.weight_decay)
        self.sklearn_model = linear_model.LogisticRegression(
            C=C,
            tol=1e-8,
            fit_intercept=False, 
            solver='lbfgs',
            # multi_class='multinomial',
            warm_start=True,
            max_iter=1000)

        C_minus_one = 1.0 / ((self.num_train_examples - 1) * self.weight_decay)
        self.sklearn_model_minus_one = linear_model.LogisticRegression(
            C=C_minus_one,
            tol=1e-8,
            fit_intercept=False, 
            solver='lbfgs',
            # multi_class='multinomial',
            warm_start=True,
            max_iter=1000)        


    def get_all_params(self):
        all_params = []
        for layer in ['softmax_linear']:
            # for var_name in ['weights', 'biases']:
            for var_name in ['weights']:                
                temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (layer, var_name))            
                all_params.append(temp_tensor)      
        return all_params        
        

    def placeholder_inputs(self):
        input_placeholder = tf.placeholder(
            tf.float32, 
            shape=(None, self.input_dim),
            name='input_placeholder')
        labels_placeholder = tf.placeholder(
            tf.int32,             
            shape=(None),
            name='labels_placeholder')
        return input_placeholder, labels_placeholder


    def fill_feed_dict_with_all_ex(self, data_set):
        feed_dict = {
            self.input_placeholder: data_set.x,
            self.labels_placeholder: data_set.labels,
            K.learning_phase(): 0
        }
        return feed_dict


    def fill_feed_dict_with_all_but_one_ex(self, data_set, idx_to_remove):
        num_examples = data_set.x.shape[0]
        idx = np.array([True] * num_examples, dtype=bool)
        idx[idx_to_remove] = False
        feed_dict = {
            self.input_placeholder: data_set.x[idx, :],
            self.labels_placeholder: data_set.labels[idx],
            K.learning_phase(): 0
        }
        return feed_dict


    def fill_feed_dict_with_batch(self, data_set, batch_size=0):
        if batch_size is None:
            return self.fill_feed_dict_with_all_ex(data_set)
        elif batch_size == 0:
            batch_size = self.batch_size
    
        input_feed, labels_feed = data_set.next_batch(batch_size)                              
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,            
            K.learning_phase(): 0
        }
        return feed_dict


    def fill_feed_dict_with_some_ex(self, data_set, target_indices):
        input_feed = data_set.x[target_indices, :].reshape(len(target_indices), -1)
        labels_feed = data_set.labels[target_indices].reshape(-1)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
            K.learning_phase(): 0            
        }
        return feed_dict


    def fill_feed_dict_with_one_ex(self, data_set, target_idx):
        input_feed = data_set.x[target_idx, :].reshape(1, -1)
        labels_feed = data_set.labels[target_idx].reshape(1)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
            K.learning_phase(): 0            
        }
        return feed_dict


    def load_inception_weights(self):
        # Replace this with a local copy for reproducibility
        # TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
        # weights_path = get_file('inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
        #                         TF_WEIGHTS_PATH_NO_TOP,
        #                         cache_subdir='models',
        #                         md5_hash='bcbd6486424b2319ff4ef7d526e38f63')
        weights_path = 'inception/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
        self.inception_model.load_weights(weights_path)


    def inference(self, input):        
        reshaped_input = tf.reshape(input, [-1, self.img_side, self.img_side, self.num_channels])
        self.inception_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=reshaped_input)
        
        raw_inception_features = self.inception_model.output

        pooled_inception_features = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(raw_inception_features)
        self.inception_features = Flatten(name='flatten')(pooled_inception_features)


        with tf.variable_scope('softmax_linear'):
            weights = variable_with_weight_decay(
                'weights', 
                [self.num_features],
                stddev=1.0 / math.sqrt(float(self.num_features)),
                wd=self.weight_decay)            
            
            logits = tf.matmul(self.inception_features, tf.reshape(weights, [-1, 1])) 
            zeros = tf.zeros_like(logits)
            logits_with_zeros = tf.concat([zeros, logits], 1)

        self.weights = weights

        return logits_with_zeros


    def predictions(self, logits):
        preds = tf.nn.softmax(logits, name='preds')
        return preds


    def set_params(self):
        # See if we can automatically infer weight shape
        self.W_placeholder = tf.placeholder(
            tf.float32,
            shape=[self.num_features],
            name='W_placeholder')
        set_weights = tf.assign(self.weights, self.W_placeholder, validate_shape=True)    
        return [set_weights]


    def retrain(self, num_steps, feed_dict):        
        self.train_with_LBFGS(
            feed_dict=feed_dict,
            save_checkpoints=False, 
            verbose=False)


    def train(self, num_steps=None, 
              iter_to_switch_to_batch=None, 
              iter_to_switch_to_sgd=None,
              save_checkpoints=True, verbose=True):

        self.train_with_LBFGS(
            feed_dict=self.all_train_feed_dict,
            save_checkpoints=save_checkpoints, 
            verbose=verbose)


    def train_with_SGD(self, **kwargs):
        super(BinaryInceptionModel, self).train(**kwargs)


    def minibatch_inception_features(self, feed_dict):

        num_examples = feed_dict[self.input_placeholder].shape[0]
        batch_size = 100        
        num_iter = int(np.ceil(num_examples / batch_size))

        ret = np.zeros([num_examples, self.num_features])

        batch_feed_dict = {}
        batch_feed_dict[K.learning_phase()] = 0

        for i in xrange(num_iter):
            start = i * batch_size
            end = (i+1) * batch_size
            if end > num_examples:
                end = num_examples

            batch_feed_dict[self.input_placeholder] = feed_dict[self.input_placeholder][start:end]
            batch_feed_dict[self.labels_placeholder] = feed_dict[self.labels_placeholder][start:end]
            ret[start:end, :] = self.sess.run(self.inception_features, feed_dict=batch_feed_dict)            
            
        return ret


    def train_with_LBFGS(self, feed_dict, save_checkpoints=True, verbose=True):
        # More sanity checks to see if predictions are the same?
                
        # X_train = feed_dict[self.input_placeholder]    
        # X_train = self.sess.run(self.inception_features, feed_dict=feed_dict)
        X_train = self.minibatch_inception_features(feed_dict)

        Y_train = feed_dict[self.labels_placeholder]
        num_train_examples = len(Y_train)
        assert len(Y_train.shape) == 1
        assert X_train.shape[0] == Y_train.shape[0]

        if num_train_examples == self.num_train_examples:
            print('Using normal model')
            model = self.sklearn_model
        elif num_train_examples == self.num_train_examples - 1:
            print('Using model minus one')
            model = self.sklearn_model_minus_one
        else:
            raise ValueError, "feed_dict has incorrect number of training examples"

        model.fit(X_train, Y_train)
        # sklearn returns coefficients in shape num_classes x num_features
        # whereas our weights are defined as num_features x num_classes
        # so we have to tranpose them first.
        W = np.reshape(model.coef_.T, -1)
        # b = model.intercept_

        params_feed_dict = {}
        params_feed_dict[self.W_placeholder] = W

        # params_feed_dict[self.b_placeholder] = b
        self.sess.run(self.set_params_op, feed_dict=params_feed_dict)
        if save_checkpoints: self.saver.save(self.sess, self.checkpoint_file, global_step=0)

        if verbose:
            print('LBFGS training took %s iter.' % model.n_iter_)
            print('After training with LBFGS: ')
            self.print_model_eval()


    def load_weights_from_disk(self, weights_filename, do_check=True, do_save=True):
        W = np.load('%s' % weights_filename)

        params_feed_dict = {}
        params_feed_dict[self.W_placeholder] = W
        self.sess.run(self.set_params_op, feed_dict=params_feed_dict)
        if do_save: self.saver.save(self.sess, self.checkpoint_file, global_step=0)  

        print('Loaded weights from disk.')
        if do_check: self.print_model_eval()


    def get_hessian(self):

        H = self.sess.run(self.hessians_op)
        print(H.shape)

        # Maybe update Hessian every time main train routine is called?









