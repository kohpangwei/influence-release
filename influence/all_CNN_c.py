from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import abc
import sys

import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, cluster
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse 

import os.path
import time
import IPython
import tensorflow as tf
import math

from influence.genericNeuralNet import GenericNeuralNet, variable, variable_with_weight_decay
from influence.dataset import DataSet

def conv2d(x, W, r):
    return tf.nn.conv2d(x, W, strides=[1, r, r, 1], padding='VALID')

def softplus(x):
    return tf.log(tf.exp(x) + 1)


class All_CNN_C(GenericNeuralNet):

    def __init__(self, input_side, input_channels, conv_patch_size, hidden1_units, hidden2_units, hidden3_units, weight_decay, **kwargs):
        self.weight_decay = weight_decay
        self.input_side = input_side
        self.input_channels = input_channels
        self.input_dim = self.input_side * self.input_side * self.input_channels
        self.conv_patch_size = conv_patch_size
        self.hidden1_units = hidden1_units
        self.hidden2_units = hidden2_units
        self.hidden3_units = hidden3_units

        super(All_CNN_C, self).__init__(**kwargs)


    def conv2d_softplus(self, input_x, conv_patch_size, input_channels, output_channels, stride):
        weights = variable_with_weight_decay(
            'weights', 
            [conv_patch_size * conv_patch_size * input_channels * output_channels],
            stddev=2.0 / math.sqrt(float(conv_patch_size * conv_patch_size * input_channels)),
            wd=self.weight_decay)
        biases = variable(
            'biases',
            [output_channels],
            tf.constant_initializer(0.0))
        weights_reshaped = tf.reshape(weights, [conv_patch_size, conv_patch_size, input_channels, output_channels])
        hidden = tf.nn.tanh(conv2d(input_x, weights_reshaped, stride) + biases)

        return hidden



    def get_all_params(self):
        all_params = []
        for layer in ['h1_a', 'h1_c', 'h2_a', 'h2_c', 'h3_a', 'h3_c', 'softmax_linear']:        
            for var_name in ['weights', 'biases']:
                temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (layer, var_name))            
                all_params.append(temp_tensor)      
        return all_params        
        

    def retrain(self, num_steps, feed_dict):        

        retrain_dataset = DataSet(feed_dict[self.input_placeholder], feed_dict[self.labels_placeholder])

        for step in xrange(num_steps):   
            iter_feed_dict = self.fill_feed_dict_with_batch(retrain_dataset)
            self.sess.run(self.train_op, feed_dict=iter_feed_dict)


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


    def inference(self, input_x):        

        input_reshaped = tf.reshape(input_x, [-1, self.input_side, self.input_side, self.input_channels])
        
        # Hidden 1
        with tf.variable_scope('h1_a'):
            h1_a = self.conv2d_softplus(input_reshaped, self.conv_patch_size, self.input_channels, self.hidden1_units, stride=1)
            
        with tf.variable_scope('h1_c'):
            h1_c = self.conv2d_softplus(h1_a, self.conv_patch_size, self.hidden1_units, self.hidden1_units, stride=2)
            
        # Hidden 2
        with tf.variable_scope('h2_a'):
            h2_a = self.conv2d_softplus(h1_c, self.conv_patch_size, self.hidden1_units, self.hidden2_units, stride=1)
            
        with tf.variable_scope('h2_c'):
            h2_c = self.conv2d_softplus(h2_a, self.conv_patch_size, self.hidden2_units, self.hidden2_units, stride=2)
            
        # Shared layers / hidden 3
        with tf.variable_scope('h3_a'):
            h3_a = self.conv2d_softplus(h2_c, self.conv_patch_size, self.hidden2_units, self.hidden3_units, stride=1)        
        
        last_layer_units = 10
        with tf.variable_scope('h3_c'):
            h3_c = self.conv2d_softplus(h3_a, 1, self.hidden3_units, last_layer_units, stride=1)
        
        h3_d = tf.reduce_mean(h3_c, axis=[1, 2])
        
        with tf.variable_scope('softmax_linear'):

            weights = variable_with_weight_decay(
                'weights', 
                [last_layer_units * self.num_classes],
                stddev=1.0 / math.sqrt(float(last_layer_units)),
                wd=self.weight_decay)            
            biases = variable(
                'biases',
                [self.num_classes],
                tf.constant_initializer(0.0))

            logits = tf.matmul(h3_d, tf.reshape(weights, [last_layer_units, self.num_classes])) + biases
            
        return logits

    def predictions(self, logits):
        preds = tf.nn.softmax(logits, name='preds')
        return preds