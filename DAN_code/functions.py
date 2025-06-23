from tensorflow.keras import backend as k
import tensorflow as tf
import numpy as np
import jax.numpy as jnp

def sqrt1pm1(x):
    return np.where(x < 1, x / ((1 + x)**(1/2) + 1), (1 + x)**(1/2) - 1)

def jax_sqrt1pm1(x):
    return jnp.where(x < 1, x / ((1 + x)**(1/2) + 1), (1 + x)**(1/2) - 1)

def log1p_sqrt1p_mlog2(x):
    return 1/2 * np.log1p(x) + np.log1p(1/(1 + x)**(1/2)) - np.log(2)

def jax_log1p_sqrt1p_mlog2(x):
    return 1/2 * jnp.log1p(x) + jnp.log1p(1/(1 + x)**(1/2)) - jnp.log(2)

### Calculate correlation(w, x) when w and x are numpy arrays
def dense_cor(x, w):
    x = x / np.sum(x**2, axis = 1, keepdims = True)**(1/2)
    
    return x @ w

### Softmax (or Boltzmann distribution) along axis
def softmax(f, tau = -np.inf, axis = None):
    c = np.max(f, axis = axis, keepdims = True)
    c = np.maximum(c, tau)
    
    m = np.exp(f - c)
    m_0 = np.exp(tau - c)
    
    return m / (np.sum(m, axis = axis, keepdims = True) + m_0)

# Weighed softmax along axis = -1
def weighed_softmax(f, g, tau = -np.inf, target_y = None, softening = 0):
    c = np.max(f, axis = -1, keepdims = True)
    c = np.maximum(c, tau)
    
    m = np.exp(f - c)
    m_0 = np.exp(tau - c)
    
    # g: (number_hidden_units + 1, number_classes)
    # m: (batch_dim, number_hidden_units)
    # prior_y: (batch_dim, number_classes)
    
    inverse_Z = 1/(m @ g[: -1] + m_0 * g[-1 :])
    
    # inverse_Z: (batch_dim, number_classes)
    # target_y: (batch_dim, number_classes)
    # g: (number_hidden_units + 1, number_classes)
    
    if target_y is not None:
        inverse_Z = np.concatenate([target_y, 1 - np.sum(target_y, axis = -1, keepdims = True)], axis = -1) * inverse_Z
    
    inverse_Z = inverse_Z @ g[: -1].T
    
    # (batch_dim, number_hidden_units) or (batch_dim, 1)
    
    return m * inverse_Z

#def log_gamma_zero(N):
#    # np.log(2) + N/2 * np.log(np.pi) - 1/2 * np.log(2 * np.pi) - 1/2 * (N + 1) * np.log(N/2 - 1) + 1/2 * N - 1
#    return 1/2 * np.log(2) + 1/2 * (N - 1) * np.log(np.pi) - 1/2 * (N + 1) * np.log(N/2 - 1) + 1/2 * N - 1

def log_gamma_ratio(beta, N):
    
    rho = beta / (N - 2)
    two_rho_squared = (2 * rho)**2
    
    eta = sqrt1pm1(two_rho_squared) - log1p_sqrt1p_mlog2(two_rho_squared)
    
    tau = -1/4 * np.log1p(two_rho_squared) + (N/2 - 1) * eta
    
    return tau

def jax_log_gamma_ratio(beta, N):
    
    rho = beta / (N - 2)
    two_rho_squared = (2 * rho)**2
    
    eta = jax_sqrt1pm1(two_rho_squared) - jax_log1p_sqrt1p_mlog2(two_rho_squared)
    
    tau = -1/4 * jnp.log1p(two_rho_squared) + (N/2 - 1) * eta
    
    return tau

def unaveraged_rayleigh_quotient(beta, h, x, w, u):
    w_u = k.sum(w * u, axis = 0)
    x_u = k.dot(x, u)
    
    q = (x_u - h * w_u)**2
    q = beta * q
    
    q = q + h * (w_u - 1) * (w_u + 1)
    beta = beta * q
    
    return q

def row_shifted_exp(tensor):
    return k.exp(tensor - k.max(tensor, axis = 1, keepdims = True))
    #def inner_row_shifted_exp():
    #    return k.exp(tensor - k.max(tensor, axis = 1, keepdims = True))
    #
    #return inner_row_shifted_exp

def col_shifted_exp(tensor):
    return k.exp(tensor - k.max(tensor, axis = 0, keepdims = True))
    #def inner_col_shifted_exp():
    #    return k.exp(tensor - k.max(tensor, axis = 0, keepdims = True))
    #
    #return inner_col_shifted_exp

def k_mins(values, k = 1):
    return -tf.math.top_k(-values, k)[0]

def kth_min(values, k = 1):
    return k_mins(values, k)[-1]