from tensorflow.keras import backend as k
import tensorflow as tf
import numpy as np
import jax.numpy as jnp

### Return the max of a tensorflow tensor in absolute value along an axis
def tensor_max_norm(tensor, axis = None, keepdims = True):
    return k.maximum(-k.min(tensor, axis = axis, keepdims = keepdims), k.max(tensor, axis = axis, keepdims = keepdims))

### Return the max of a numpy array in absolute value along an axis.
def array_max_norm(array, axis = None, keepdims = True):
    return np.maximum(-np.min(array, axis = axis, keepdims = keepdims), np.max(array, axis = axis, keepdims = keepdims))

### Return the max of a jax array in absolute value along an axis.
def jax_max_norm(array, axis = None, keepdims = True):
    return jnp.maximum(-jnp.min(array, axis = axis, keepdims = keepdims), jnp.max(array, axis = axis, keepdims = keepdims))

def tensor_two_norm(tensor, axis = None, keepdims = True):
    return k.sum(tensor**2, axis = axis, keepdims = keepdims)**(1/2)

def array_two_norm(array, axis = None, keepdims = True):
    return np.sum(array**2, axis = axis, keepdims = keepdims)**(1/2)

def jax_two_norm(array, axis = None, keepdims = True):
    return jnp.sum(array**2, axis = axis, keepdims = keepdims)**(1/2)

def tensor_one_norm(tensor, axis = None, keepdims = True):
    return k.sum(k.abs(tensor), axis = axis, keepdims = keepdims)

def array_one_norm(array, axis = None, keepdims = True):
    return np.sum(np.abs(array), axis = axis, keepdims = keepdims)

def jax_one_norm(array, axis = None, keepdims = True):
    return jnp.sum(jnp.abs(array), axis = axis, keepdims = keepdims)

def tensor_log_one_norm(tensor, log_gap, axis = None, keepdims = True):
    maximum = k.max(tensor, axis = axis, keepdims = keepdims)
    maximum = k.maximum(tensor, log_gap)
    return maximum + k.log(k.sum(k.exp(tensor - maximum), axis = axis, keepdims = keepdims) + k.exp(log_gap - maximum))

def array_log_one_norm(array, log_gap, axis = None, keepdims = True):
    maximum = np.max(tensor, axis = axis, keepdims = keepdims)
    maximum = np.maximum(tensor, log_gap)
    return maximum + np.log(np.sum(np.exp(array - maximum), axis = axis, keepdims = keepdims) + np.exp(log_gap - maximum))

### Normalize a tensorflow tensor of vectors by their norms, returning unit vectors corresponding to vector directions
def tensor_normalize(tensor, norms, sub_value = 0.):
    
    directions = tensor / norms
    
    # If directions are nan we replace them by 0 because a vector with norm 0 has no direction
    directions = tf.where(tf.math.is_nan(directions), sub_value, directions)
    
    return directions

### Normalize a numpy array of vectors by their norms, returning unit vectors corresponding to vector directions
def array_normalize(array, norms, sub_value = 0.):
    
    directions = array / norms
    
    # If directions are nan we replace them by 0 because a vector with norm 0 has no direction
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

#### Normalize the exp of vectors by their norms, returning unit vectors corresponding to vector directions
#def tensor_log_normalize(tensor, norms):
#    
#    log_directions = tensor - norms
#    
#    # If log_directions are nan we replace them by -infinity because a vector with norm 0 has no direction
#    log_directions = tf.where(tf.math.is_nan(log_directions), -np.inf, log_directions)
#    
#    return log_directions
#
#### Normalize the exp of vectors by their norms, returning unit vectors corresponding to vector directions
#def array_log_normalize(array, norms):
#    
#    log_directions = array - norms
#    
#    # If log_directions are nan we replace them by -infinity because a vector with norm 0 has no direction
#    log_directions = np.where(np.is_nan(log_directions), -np.inf, log_directions)
#    
#    return log_directions