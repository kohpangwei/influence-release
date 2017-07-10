import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K


sns.set(color_codes=True)

def plot_flat_bwimage(X, y=None, pos_class=1, neg_class=-1, side=28):
    X = np.reshape(X, (side, side))
    if y is not None:
        if y == 1:
            label = pos_class
        else:
            label = neg_class
    with sns.axes_style("white"):
        if y is not None:
            plt.title('Label is %s' % label)
        plt.imshow(X, cmap='gray', interpolation='none')

def plot_flat_bwgrad(X, side=28):
    X = np.reshape(X, (side, side))
    max_abs = np.max(np.abs(X))
    with sns.axes_style("white"):
        f, ax = plt.subplots()
        colormap = ax.imshow(X, cmap='coolwarm', vmax=max_abs, vmin=-max_abs, interpolation='none')
        f.colorbar(colormap)

def plot_flat_colorimage(X, y, pos_class=1, neg_class=-1, side=32):
    X = np.reshape(X, (side, side, 3))
    if y == 1:
        label = pos_class
    else:
        label = neg_class
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(6,6))
        ax.set_title('Label is %s' % label)        
        ax.imshow(X, interpolation='none')  
        # ax.imshow(X)        
        plt.show()

def plot_flat_colorgrad(X, side=32):
    X = np.reshape(X, (side, side, 3))
    with sns.axes_style("white"):
        f, ax = plt.subplots()
        colormap = ax.imshow(X, interpolation='none')
        f.colorbar(colormap)
