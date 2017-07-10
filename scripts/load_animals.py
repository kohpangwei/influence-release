import os

from tensorflow.contrib.learn.python.learn.datasets import base
import numpy as np
import IPython

from subprocess import call

from keras.preprocessing import image

from influence.dataset import DataSet
from influence.inception_v3 import preprocess_input

BASE_DIR = 'data' # TODO: change

def fill(X, Y, idx, label, img_path, img_side):
    img = image.load_img(img_path, target_size=(img_side, img_side))
    x = image.img_to_array(img)
    X[idx, ...] = x
    Y[idx] = label

     
def extract_and_rename_animals():
    class_maps = [
        ('dog', 'n02084071'),
        ('cat', 'n02121808'),
        ('bird', 'n01503061'),
        ('fish', 'n02512053'),
        ('horse', 'n02374451'),
        ('monkey', 'n02484322'),
        ('zebra', 'n02391049'),
        ('panda', 'n02510455'),
        ('lemur', 'n02496913'),
        ('wombat', 'n01883070'),
        ]


    for class_string, class_id in class_maps:
        
        class_dir = os.path.join(BASE_DIR, class_string)
        print(class_dir)
        call('mkdir %s' % class_dir, shell=True)
        call('tar -xf %s.tar -C %s' % (os.path.join(BASE_DIR, class_id), class_dir), shell=True)
        
        for filename in os.listdir(class_dir):

            file_idx = filename.split('_')[1].split('.')[0]
            src_filename = os.path.join(class_dir, filename)
            dst_filename = os.path.join(class_dir, '%s_%s.JPEG' % (class_string, file_idx))
            os.rename(src_filename, dst_filename)



def load_animals(num_train_ex_per_class=300, 
                 num_test_ex_per_class=100,
                 num_valid_ex_per_class=0,
                 classes=None,
                 ):    

    num_channels = 3
    img_side = 299

    if num_valid_ex_per_class == 0:
        valid_str = ''
    else:
        valid_str = '_valid-%s' % num_valid_examples

    if classes is None:
        classes = ['dog', 'cat', 'bird', 'fish', 'horse', 'monkey', 'zebra', 'panda', 'lemur', 'wombat']
        data_filename = os.path.join(BASE_DIR, 'dataset_train-%s_test-%s%s.npz' % (num_train_ex_per_class, num_test_ex_per_class, valid_str))
    else:
        data_filename = os.path.join(BASE_DIR, 'dataset_%s_train-%s_test-%s%s.npz' % ('-'.join(classes), num_train_ex_per_class, num_test_ex_per_class, valid_str))

    num_classes = len(classes)
    num_train_examples = num_train_ex_per_class * num_classes
    num_test_examples = num_test_ex_per_class * num_classes
    num_valid_examples = num_valid_ex_per_class * num_classes

    if os.path.exists(data_filename):
        print('Loading animals from disk...')
        f = np.load(data_filename)
        X_train = f['X_train']
        X_test = f['X_test']
        Y_train = f['Y_train']
        Y_test = f['Y_test']

        if 'X_valid' in f:
            X_valid = f['X_valid']
        else:
            X_valid = None

        if 'Y_valid' in f:
            Y_valid = f['Y_valid']
        else:
            Y_valid = None

    else:
        print('Reading animals from raw images...')
        X_train = np.zeros([num_train_examples, img_side, img_side, num_channels])
        X_test = np.zeros([num_test_examples, img_side, img_side, num_channels])
        X_valid = np.zeros([num_valid_examples, img_side, img_side, num_channels])

        Y_train = np.zeros([num_train_examples])
        Y_test = np.zeros([num_test_examples])
        Y_valid = np.zeros([num_valid_examples])

        for class_idx, class_string in enumerate(classes):
            print('class: %s' % class_string)            
            # For some reason, a lot of numbers are skipped.
            i = 0
            num_filled = 0
            while num_filled < num_train_ex_per_class:        
                img_path = os.path.join(BASE_DIR, '%s/%s_%s.JPEG' % (class_string, class_string, i))
                print(img_path)
                if os.path.exists(img_path):
                    fill(X_train, Y_train, num_filled + (num_train_ex_per_class * class_idx), class_idx, img_path, img_side)
                    num_filled += 1
                    print(num_filled)
                i += 1

            num_filled = 0
            while num_filled < num_test_ex_per_class:        
                img_path = os.path.join(BASE_DIR, '%s/%s_%s.JPEG' % (class_string, class_string, i))
                if os.path.exists(img_path):
                    fill(X_test, Y_test, num_filled + (num_test_ex_per_class * class_idx), class_idx, img_path, img_side)
                    num_filled += 1
                    print(num_filled)
                i += 1

            num_filled = 0
            while num_filled < num_valid_ex_per_class:        
                img_path = os.path.join(BASE_DIR, '%s/%s_%s.JPEG' % (class_string, class_string, i))
                if os.path.exists(img_path):
                    fill(X_valid, Y_valid, num_filled + (num_valid_ex_per_class * class_idx), class_idx, img_path, img_side)
                    num_filled += 1
                    print(num_filled)
                i += 1

        X_train = preprocess_input(X_train)
        X_test = preprocess_input(X_test)
        X_valid = preprocess_input(X_valid)

        np.random.seed(0)
        permutation_idx = np.arange(num_train_examples)
        np.random.shuffle(permutation_idx)
        X_train = X_train[permutation_idx, :]
        Y_train = Y_train[permutation_idx]
        permutation_idx = np.arange(num_test_examples)
        np.random.shuffle(permutation_idx)
        X_test = X_test[permutation_idx, :]
        Y_test = Y_test[permutation_idx]
        permutation_idx = np.arange(num_valid_examples)
        np.random.shuffle(permutation_idx)
        X_valid = X_valid[permutation_idx, :]
        Y_valid = Y_valid[permutation_idx]

        np.savez_compressed(data_filename, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, X_valid=X_valid, Y_valid=Y_valid)


    train = DataSet(X_train, Y_train)
    if (X_valid is not None) and (Y_valid is not None):
        validation = DataSet(X_valid, Y_valid)
    else:
        validation = None

    test = DataSet(X_test, Y_test)

    return base.Datasets(train=train, validation=validation, test=test)


def load_koda():
    num_channels = 3
    img_side = 299

    data_filename = os.path.join(BASE_DIR, 'dataset_koda.npz')

    if os.path.exists(data_filename):
        print('Loading Koda from disk...')
        f = np.load(data_filename)
        X = f['X']
        Y = f['Y']
    else:
        # Returns all class 0
        print('Reading Koda from raw images...')

        image_files = [image_file for image_file in os.listdir(os.path.join(BASE_DIR, 'koda')) if (image_file.endswith('.jpg'))]
        # Hack to get the image files in the right order
        # image_files = [image_file for image_file in os.listdir(os.path.join(BASE_DIR, 'koda')) if (image_file.endswith('.jpg') and not image_file.startswith('124'))]
        # image_files += [image_file for image_file in os.listdir(os.path.join(BASE_DIR, 'koda')) if (image_file.endswith('.jpg') and image_file.startswith('124'))]


        num_examples = len(image_files)
        X = np.zeros([num_examples, img_side, img_side, num_channels])
        Y = np.zeros([num_examples])

        class_idx = 0
        for counter, image_file in enumerate(image_files):
            img_path = os.path.join(BASE_DIR, 'koda', image_file)
            fill(X, Y, counter, class_idx, img_path, img_side)

        X = preprocess_input(X)

        np.savez(data_filename, X=X, Y=Y)

    return X, Y
    

def load_dogfish_with_koda():        
    classes = ['dog', 'fish']
    X_test, Y_test = load_koda()

    data_sets = load_animals(num_train_ex_per_class=900, 
                 num_test_ex_per_class=300,
                 num_valid_ex_per_class=0,
                 classes=classes)
    train = data_sets.train
    validation = data_sets.validation
    test = DataSet(X_test, Y_test)

    return base.Datasets(train=train, validation=validation, test=test)


def load_dogfish_with_orig_and_koda():
    classes = ['dog', 'fish']
    X_test, Y_test = load_koda()
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    data_sets = load_animals(num_train_ex_per_class=900, 
                 num_test_ex_per_class=300,
                 num_valid_ex_per_class=0,
                 classes=classes)
    train = data_sets.train
    validation = data_sets.validation

    test = DataSet(
        np.concatenate((data_sets.test.x, X_test), axis=0), 
        np.concatenate((data_sets.test.labels, Y_test), axis=0))

    return base.Datasets(train=train, validation=validation, test=test)

