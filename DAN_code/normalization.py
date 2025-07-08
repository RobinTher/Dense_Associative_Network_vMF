from tensorflow.keras import backend as k
import tensorflow as tf
import numpy as np
import jax.numpy as jnp

### Return the max of a tensorflow tensor in absolute value along an axis
def tensor_max_norm(tensor, axis = None, keepdims = True):
    '''
    Compute the maximum absolute value of a tensor along a specified axis.

    Parameters
    ----------
    tensor : tf.Tensor
        A tensor.
    axis : int, optional
        The axis along which to compute the maximum absolute value. If None, the norm
        is computed over all elements of the tensor.
    keepdims : bool, optional
        If True, the reduced dimensions are retained with size 1. Default is True.
    Returns
    -------
    tf.Tensor
        A tensor containing the maximum absolute value of the input tensor along
        the specified axis.
    '''
    return k.maximum(-k.min(tensor, axis = axis, keepdims = keepdims), k.max(tensor, axis = axis, keepdims = keepdims))

### Return the max of a numpy array in absolute value along an axis.
def array_max_norm(array, axis = None, keepdims = True):
    '''
    Compute the maximum absolute value of a numpy array along a specified axis.

    Parameters
    ----------
    array : np.ndarray
        A numpy array.
    axis : int, optional
        The axis along which to compute the maximum absolute value. If None, the norm
        is computed over all elements of the array.
    keepdims : bool, optional
        If True, the reduced dimensions are retained with size 1. Default is True.
    Returns
    -------
    np.ndarray
        A numpy array containing the maximum absolute value of the input array along
        the specified axis.
    '''
    return np.maximum(-np.min(array, axis = axis, keepdims = keepdims), np.max(array, axis = axis, keepdims = keepdims))

### Return the max of a jax array in absolute value along an axis.
def jax_max_norm(array, axis = None, keepdims = True):
    return jnp.maximum(-jnp.min(array, axis = axis, keepdims = keepdims), jnp.max(array, axis = axis, keepdims = keepdims))

def tensor_two_norm(tensor, axis = None, keepdims = True):
    '''
    Compute the two-norm of a tensor along a specified axis.

    Parameters
    ----------
    tensor : tf.Tensor
        A tensor.
    axis : int, optional
        The axis along which to compute the two-norm. If None, the norm is computed
        over all elements of the tensor.
    keepdims : bool, optional
        If True, the reduced dimensions are retained with size 1. Default is True.
    Returns
    -------
    tf.Tensor
        A tensor containing the two-norm of the input tensor along the specified axis.
    '''
    return k.sum(tensor**2, axis = axis, keepdims = keepdims)**(1/2)

def array_two_norm(array, axis = None, keepdims = True):
    '''
    Compute the two-norm of a numpy array along a specified axis.

    Parameters
    ----------
    array : np.ndarray
        A numpy array.
    axis : int, optional
        The axis along which to compute the two-norm. If None, the norm is computed
        over all elements of the array.
    keepdims : bool, optional
        If True, the reduced dimensions are retained with size 1. Default is True.
    Returns
    -------
    np.ndarray
        A numpy array containing the two-norm of the input array along the specified axis.
    '''
    return np.sum(array**2, axis = axis, keepdims = keepdims)**(1/2)

def jax_two_norm(array, axis = None, keepdims = True):
    return jnp.sum(array**2, axis = axis, keepdims = keepdims)**(1/2)

def tensor_one_norm(tensor, axis = None, keepdims = True):
    '''
    Compute the one-norm of a tensor along a specified axis.

    Parameters
    ----------
    tensor : tf.Tensor
        A tensor.
    axis : int, optional
        The axis along which to compute the one-norm. If None, the norm is computed
        over all elements of the tensor.
    keepdims : bool, optional
        If True, the reduced dimensions are retained with size 1. Default is True.
    Returns
    -------
    tf.Tensor
        A tensor containing the one-norm of the input tensor along the specified axis.
    '''
    return k.sum(k.abs(tensor), axis = axis, keepdims = keepdims)

def array_one_norm(array, axis = None, keepdims = True):
    '''
    Compute the one-norm of a numpy array along a specified axis.

    Parameters
    ----------
    array : np.ndarray
        A numpy array.
    axis : int, optional
        The axis along which to compute the one-norm. If None, the norm is computed
        over all elements of the array.
    keepdims : bool, optional
        If True, the reduced dimensions are retained with size 1. Default is True.
    Returns
    -------
    np.ndarray
        A numpy array containing the one-norm of the input array along the specified axis.
    '''
    return np.sum(np.abs(array), axis = axis, keepdims = keepdims)

def jax_one_norm(array, axis = None, keepdims = True):
    return jnp.sum(jnp.abs(array), axis = axis, keepdims = keepdims)

def tensor_normalize(tensor, norms, sub_value = 0.):
    '''
    Normalize a tensor of vectors by their norms, returning unit vectors corresponding
    to vector directions. If a vector has a norm of 0, its direction is set to a specified
    substitute value (default is 0).

    Parameters
    ----------
    tensor : tf.Tensor
        A tensor of vectors to be normalized.
    norms : tf.Tensor
        A tensor containing the norms of the vectors in the input tensor.
    sub_value : float, optional
        A value to substitute for directions of vectors with a norm of 0. Default is 0.
    Returns
    -------
    tf.Tensor
        A tensor containing the normalized vectors.
    '''
    directions = tensor / norms
    
    directions = tf.where(tf.math.is_nan(directions), sub_value, directions)
    
    return directions

def array_normalize(array, norms, sub_value = 0.):
    '''
    Normalize a numpy array of vectors by their norms, returning unit vectors corresponding
    to vector directions. If a vector has a norm of 0, its direction is set to a specified
    substitute value (default is 0).

    Parameters
    ----------
    array : np.ndarray
        A numpy array of vectors to be normalized.
    norms : np.ndarray
        A numpy array containing the norms of the vectors in the input array.
    sub_value : float, optional
        A value to substitute for directions of vectors with a norm of 0. Default is 0.
    Returns
    -------
    np.ndarray
        A numpy array containing the normalized vectors.
    '''
    directions = array / norms
    
    directions = np.where(np.isnan(directions), sub_value, directions)
    
    return directions

### Normalize a numpy array of vectors by their norms, returning unit vectors corresponding to vector directions
def jax_normalize(array, norms, sub_value = 0.):
    
    directions = array / norms
    
    # If directions are nan we replace them by 0 because a vector with norm 0 has no direction
    directions = jnp.where(jnp.isnan(directions), sub_value, directions)
    
    return directions

def tensor_subsphere_normalize(tensor, center, radiuses):
    # Change coordinates such that the center is at the origin
    # tensor = tensor - (1 - 1/2*radiuses**2) * center
    tensor = tensor - (1 - 1/np.sqrt(2)*radiuses)*(1 + 1/np.sqrt(2)*radiuses) * center
    
    # Project onto the hyperplane containing the subsphere
    tensor = tensor - k.sum(center * tensor, axis = 1, keepdims = True) * center
    
    # Project onto the subsphere
    tensor = ((1 - radiuses/2)*(1 + radiuses/2))**(1/2)*radiuses * tensor / tensor_two_norm(tensor, axis = 1)
    
    # Change coordinates back
    tensor = tensor + (1 - 1/np.sqrt(2)*radiuses)*(1 + 1/np.sqrt(2)*radiuses) * center
    
    return tensor