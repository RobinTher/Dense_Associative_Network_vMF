import tensorflow as tf
import numpy as np

import DAN_code.normalization as norm

def prepare_flat_data(dataset, normalize_data = True, softening = None):
    
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    
    x_train = np.reshape(x_train, (x_train.shape[0], -1)) / 255
    x_test = np.reshape(x_test, (x_test.shape[0], -1)) / 255
    
    if normalize_data:
        x_train = x_train - np.mean(x_train, axis = 1, keepdims = True)
        x_train = norm.array_normalize(x_train, norm.array_two_norm(x_train, axis = 1))
        
        x_test = x_test - np.mean(x_test, axis = 1, keepdims = True)
        x_test = norm.array_normalize(x_test, norm.array_two_norm(x_test, axis = 1))
    else:
        # Round to the nearest 8-bit representation
        x_train = np.rint(x_train * 2**8) / 2**8
        x_test = np.rint(x_test * 2**8) / 2**8
    
    y_train = tf.one_hot(y_train, 10)
    y_test = tf.one_hot(y_test, 10)
    
    if softening is not None:
        y_train = (1 - softening) * y_train + softening / (y_train.shape[1] + 1)
    
    return x_train, y_train, x_test, y_test