import tensorflow as tf
import numpy as np

from tensorflow.image import extract_patches

import DAN_code.normalization as norm

def prepare_flat_data(dataset, normalize_data = True, softening = None):
    '''
    Prepare flat data for training and testing.
    This function loads the dataset, flattens the data into vectors,
    normalizes it unless specified otherwise,
    and applies softening to the labels if specified.

    Parameters
    ----------
    dataset : tensorflow.keras.datasets module
        The dataset to load and process.
    normalize_data : bool, optional
        Whether to normalize the data. Default is True.
    softening : float, optional
        Label softening or smoothing. If None, no softening is applied.

    Returns
    -------
    x_train : np.ndarray
        The training data reshaped into vectors and normalized.
    y_train : np.ndarray
        The training labels one-hot encoded and softened if specified.
    x_test : np.ndarray
        The testing data reshaped into vectors and normalized.
    y_test : np.ndarray
        The testing labels one-hot encoded and softened if specified.
    '''
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

def prepare_patch_data(dataset, normalize_data = True):
    '''
    Prepare patch data for training and testing.
    This function loads the dataset, extracts patches from it,
    flattens them into vectors and normalizes them unless specified otherwise.

    Parameters
    ----------
    dataset : tensorflow.keras.datasets module
        The dataset to load and process.
    normalize_data : bool, optional
        Whether to normalize the data. Default is True.
    Returns
    -------
    x_train : np.ndarray
        The training data reshaped into vectors and normalized.
    y_train : np.ndarray
        Placeholder training labels set to zeros.
    x_test : np.ndarray
        The testing data reshaped into vectors and normalized.
    y_test : np.ndarray
        Placeholder testing labels set to zeros.
    '''
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    
    x_train = np.reshape(x_train, (*x_train.shape, 1)) / 255
    x_test = np.reshape(x_test, (*x_test.shape, 1)) / 255
    
    x_train = extract_patches(x_train, sizes = [1, 6, 6, 1], strides = [1, 2, 2, 1], rates = [1, 1, 1, 1], padding = "VALID")
    x_test = extract_patches(x_test, sizes = [1, 6, 6, 1], strides = [1, 2, 2, 1], rates = [1, 1, 1, 1], padding = "VALID")
    
    x_train = np.reshape(x_train, (-1, x_train.shape[-1]))
    x_test = np.reshape(x_test, (-1, x_test.shape[-1]))
    
    if normalize_data:
        x_train = x_train - np.mean(x_train, axis = 1, keepdims = True)
        x_train = norm.array_normalize(x_train, norm.array_two_norm(x_train, axis = 1))
        
        x_test = x_test - np.mean(x_test, axis = 1, keepdims = True)
        x_test = norm.array_normalize(x_test, norm.array_two_norm(x_test, axis = 1))
    else:
        # Round to the nearest 8-bit representation
        x_train = np.rint(x_train * 2**8) / 2**8
        x_test = np.rint(x_test * 2**8) / 2**8
    
    y_train = np.zeros(x_train.shape[0])
    y_test = np.zeros(x_test.shape[0])
    
    return x_train, y_train, x_test, y_test