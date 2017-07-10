import IPython
import numpy as np

import os

import influence.experiments as experiments
from influence.inceptionModel import BinaryInceptionModel
from influence.dataset import DataSet

from tensorflow.contrib.learn.python.learn.datasets import base

from load_animals import load_animals, load_dogfish_with_koda

img_side = 299
num_channels = 3
 
batch_size = 100
initial_learning_rate = 0.001 
keep_probs = None
decay_epochs = [1000, 10000]
weight_decay = 0.001
# weight_decay = 0.01

### DogFish
# num_train_ex_per_class = 900
# num_test_ex_per_class = 300
# dataset_name = 'dogfish_%s_%s' % (num_train_ex_per_class, num_test_ex_per_class)
# data_sets = load_animals(
#     num_train_ex_per_class=num_train_ex_per_class, 
#     num_test_ex_per_class=num_test_ex_per_class,
#     classes=['dog', 'fish'])

### DogFish with Koda
dataset_name = 'dogfish_koda'
data_sets = load_dogfish_with_koda()

model_name = '%s_inception' % dataset_name

num_classes = 2
model = BinaryInceptionModel(
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
    model_name=dataset_name)


data_sets.train.reset_batch()
data_sets.test.reset_batch()


for data_set, label in [
    (data_sets.train, 'train'),
    (data_sets.test, 'test')]:

    data_set.reset_batch()

    num_examples = data_set.num_examples
    if num_examples > 100:
        batch_size = 100
    else:
        batch_size = num_examples
    
    assert num_examples % batch_size == 0
    num_iter = int(num_examples / batch_size)

    inception_features_val = []
    for i in xrange(num_iter):
        feed_dict = model.fill_feed_dict_with_batch(data_set, batch_size=batch_size)
        inception_features_val_temp = model.sess.run(model.inception_features, feed_dict=feed_dict)
        inception_features_val.append(inception_features_val_temp)

    np.savez(
        'data/%s_features_new_%s.npz' % (model_name, label), 
        inception_features_val=np.concatenate(inception_features_val),
        labels=data_set.labels)