from tensorflow.keras import backend as k
import tensorflow as tf
import numpy as np
# import scipy.special as spec

from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant, RandomNormal, Orthogonal
from tensorflow.keras.saving import deserialize_keras_object

import DAN_code.initializers as init
import DAN_code.constraints as constr
import DAN_code.normalization as norm
import DAN_code.functions as func

eps = np.finfo(np.float32).eps
logzero = -np.float32(151*np.log(2))
# eps = np.float32(eps**(1/2))
# eps = 0

class Normalize(Layer):
    def __init__(self, normalize_online, **kwargs):
        super(Normalize, self).__init__(**kwargs)
        self.normalize_online = normalize_online
    
    def call(self, x):
        
        if self.normalize_online:
            x = x - k.mean(x, axis = 1, keepdims = True)
            x = norm.tensor_normalize(x, norm.tensor_two_norm(x, axis = 1))
        
        return x
    
    # To support serialization
    def get_config(self):
        config = super(Normalize, self).get_config()
        config.update({"normalize_online" : self.normalize_online})
        return config

### Layer that computes w @ x using a dense dot product
class DenseNormalDot(Layer):
    
    def __init__(self, output_size, max_output_size, beta_init, **kwargs):
        super(DenseNormalDot, self).__init__(**kwargs)
        
        self.output_size = output_size

        self.max_output_size = max_output_size
        
        self.beta_init = beta_init
    
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name = "memory_kernel",
                                      shape = (input_shape[1], self.max_output_size),
                                      initializer = init.RandomNormal(self.output_size),
                                      trainable = True)
        
        self.eigvecs = self.add_weight(name = "eigvec_kernel",
                                       shape = (input_shape[1], self.max_output_size),
                                       initializer = init.RandomSpherical(self.output_size),
                                       constraint = constr.UnitTwoNorm(self.output_size, axis = 0),
                                       trainable = True)
        
        self.beta = self.add_weight(name = "beta", shape = (),
                                    initializer = Constant(self.beta_init),
                                    trainable = False)
    
    def call(self, x, training = None):
        
        @tf.custom_gradient
        def stop_activation(activation):
            def grad(stream):
                return stream
            
            return 0 * activation, grad
        
        kernel = self.kernel[:, : self.output_size]
        eigvecs = self.eigvecs[:, : self.output_size]
        
        h = self.beta * k.dot(x, kernel) - 1/2 * self.beta * k.sum(kernel**2, axis = 0, keepdims = True)
        
        q = stop_activation(self.beta**2 * (k.dot(x, eigvecs) - k.sum(k.stop_gradient(kernel) * eigvecs, axis = 0))**2 - self.beta)
        
        return [h + tf.math.log1p(q), -1/2 * self.beta * k.sum(x**2, axis = 1, keepdims = True)]
    
    # To support serialization
    def get_config(self):
        config = super(DenseNormalDot, self).get_config()
        config.update({"output_size" : self.output_size, "max_output_size" : self.max_output_size,
                       "beta_init" : self.beta_init})
        return config
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)

### Layer that computes correlation(w, x) using a dense dot product
class DenseCor(Layer):
    
    def __init__(self, output_size, max_output_size, beta_init, **kwargs):
        super(DenseCor, self).__init__(**kwargs)
        
        self.output_size = output_size

        self.max_output_size = max_output_size
        
        self.beta_init = beta_init
    
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # Try w = q c
        self.kernel = self.add_weight(name = "memory_kernel",
                                      shape = (input_shape[1], self.max_output_size),
                                      initializer = init.RandomSpherical(self.output_size),
                                      constraint = constr.UnitTwoNorm(self.output_size, axis = 0),
                                      trainable = True)
        
        self.eigvecs = self.add_weight(name = "eigvec_kernel",
                                       shape = (input_shape[1], self.max_output_size),
                                       initializer = init.RandomSpherical(self.output_size),
                                       constraint = constr.UnitTwoNorm(self.output_size, axis = 0),
                                       trainable = True)
        
        self.beta = self.add_weight(name = "beta", shape = (),
                                    initializer = Constant(self.beta_init),
                                    trainable = False)
    
    def call(self, x, training = None):
        
        @tf.custom_gradient
        def silent_normalization(kernel):
            def grad(upstream):
                downstream = upstream - k.sum(upstream * kernel, axis = 0) * kernel
                return downstream
            
            return kernel, grad
        
        @tf.custom_gradient
        def stop_activation(activation):
            def grad(stream):
                return stream
            
            return 0 * activation, grad
        
        kernel = silent_normalization(self.kernel[:, : self.output_size])
        eigvecs = self.eigvecs[:, : self.output_size]
        
        h = k.dot(x, kernel)
        
        q = stop_activation(func.unaveraged_rayleigh_quotient(self.beta, k.stop_gradient(h), x, k.stop_gradient(kernel), eigvecs))
        
        return self.beta * h + tf.math.log1p(q)
    
    # To support serialization
    def get_config(self):
        config = super(DenseCor, self).get_config()
        config.update({"output_size" : self.output_size, "max_output_size" : self.max_output_size,
                       "beta_init" : self.beta_init})
        return config
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)

class DenseOrth(Layer):
    
    def __init__(self, max_input_size, output_size, max_output_size, **kwargs):
        super(DenseOrth, self).__init__(**kwargs)
        
        self.max_input_size = max_input_size
        
        self.output_size = output_size

        self.max_output_size = max_output_size
    
    def build(self, input_shape):
        self.input_size = input_shape[1]
        # Create a trainable weight variable for this layer.
        # Try w = q c
        self.kernel = self.add_weight(name = "basis_kernel",
                                      shape = (self.max_input_size, self.max_output_size),
                                      initializer = Orthogonal(),
                                      #constraint = constr.Orthogonal(self.input_size, self.output_size),
                                      trainable = True)
        
    def call(self, x, training = None):
        
        @tf.custom_gradient
        def silent_normalization(kernel):
            def grad(upstream):
                reg = k.dot(kernel, tf.transpose(upstream))
                downstream = upstream - k.dot(kernel, (reg + tf.transpose(reg))/2)
                return downstream
            
            return kernel, grad
        
        kernel = silent_normalization(self.kernel[: self.input_size, : self.output_size])
        
        return k.dot(x, kernel)
    
    # To support serialization
    def get_config(self):
        config = super(DenseOrth, self).get_config()
        config.update({"output_size" : self.output_size, "max_output_size" : self.max_output_size})
        return config
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)

### Layer that computes log(correlation(w, x)) using a dense dot product
class LogDenseCor(Layer):

    def __init__(self, number_memories, w_mean, noise, **kwargs):
        self.number_memories = number_memories
        self.w_mean = w_mean
        self.noise = noise
        super(LogDenseCor, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name = "kernel",
                                      shape = (input_shape[1], self.number_memories),
                                      initializer = init.ClusteredNormal(self.w_mean, self.noise, 1, axis = (0,)),
                                      constraint = constr.UnitTwoNorm(axis = 0),
                                      trainable = True)
        self.kernel.norm_order = "two"
        # Axes occupied by individual feature detectors
        self.kernel.axis = (0,)
        super(LogDenseCor, self).build(input_shape)
    
    def call(self, x):
        
        x = x - k.mean(x, axis = 1, keepdims = True)
        x_var = k.sum(x**2, axis = 1, keepdims = True)
        x_is_blank = x_var == 0.
        
        # w = self.kernel / k.stop_gradient(k.maximum(k.max(self.kernel, axis = 0, keepdims = True), -k.min(self.kernel, axis = 0, keepdims = True)))
        h = k.log(tf.where(x_is_blank, 1., k.abs(k.dot(x, self.kernel))))
        h = h - (k.log(k.sum(self.kernel**2, axis = 0, keepdims = True))
                 + k.log(tf.where(x_is_blank, 1., x_var))) / 2
        
        return tf.where(x_is_blank, logzero, h)
    
    # To support serialization
    def get_config(self):
        base_config = super(LogDenseCor, self).get_config()
        return {**base_config, "number_memories" : self.number_memories, "w_mean" : self.w_mean, "noise" : self.noise}
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.number_memories)

### Layer that computes log(dot(g, exp(beta*h))) using a dense dot product
class LogDenseNormalExp(Layer):
    
    def __init__(self, number_constraint_iterations, max_input_size,
                 output_size, prior_y = None, **kwargs):
        super(LogDenseNormalExp, self).__init__(**kwargs)
        
        self.number_constraint_iterations = number_constraint_iterations
        
        self.max_input_size = max_input_size

        self.output_size = output_size
        
        if prior_y is None:
            self.prior_y = tf.Variable(tf.ones((output_size,)) / output_size, trainable = False)
        else:
            self.prior_y = tf.Variable(tf.convert_to_tensor(prior_y, dtype = "float32"),
                                       trainable = False)
    
    def build(self, input_shape):
        self.input_size = input_shape[0][1]
        
        self.counts_memory = self.add_weight(name = "count_kernel",
                                             shape = (self.max_input_size + 1, 1),
                                             initializer = "ones",
                                             trainable = False)
        
        self.counts_memory[self.input_size].assign(self.input_size/self.max_input_size)
        #self.counts_memory[self.input_size].assign(0)
        
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name = "weigh_kernel",
                                      shape = (self.max_input_size + 1, self.output_size + 1),
                                      initializer = init.Categorical(self.prior_y, self.input_size),
                                      constraint = constr.AltOneNorm(self.input_size,
                                                                     self.prior_y, self.counts_memory,
                                                                     self.number_constraint_iterations),
                                      trainable = True)
    
    def call(self, h):
        
        @tf.custom_gradient
        def silent_normalization(kernel):
            def grad(upstream):
                
                downstream = upstream - k.sum(upstream * kernel, axis = 1, keepdims = True) / self.counts_memory[: self.input_size + 1]
                
                return downstream
            
            return kernel, grad
        
        kernel = silent_normalization(self.kernel[: self.input_size + 1])
        
        c = k.stop_gradient(k.max(h[0], axis = 1, keepdims = True))
        c = k.stop_gradient(k.maximum(c, 0))
        
        return c + k.log(k.dot(k.exp(h[0] - c), kernel[: -1]) + k.exp(0 - c) * kernel[-1 :]) + h[1]
    
    # To support serialization
    def get_config(self):
        config = super(LogDenseNormalExp, self).get_config()
        config.update({"number_constraint_iterations" : self.number_constraint_iterations,
                       "max_input_size" : self.max_input_size, "output_size" : self.output_size,
                       "prior_y" : self.prior_y.value()})
        return config

### Layer that computes log(dot(g, exp(beta*h))) using a dense dot product
class LogDenseExp(Layer):
    
    def __init__(self, number_constraint_iterations, max_input_size,
                 output_size, tau_init, prior_y = None, **kwargs):
        super(LogDenseExp, self).__init__(**kwargs)
        
        self.number_constraint_iterations = number_constraint_iterations
        
        self.max_input_size = max_input_size

        self.output_size = output_size
        
        self.tau_init = tau_init
        
        if prior_y is None:
            #self.prior_y_init = np.ones((output_size,)) / output_size
            self.prior_y = tf.Variable(tf.ones((output_size,)) / output_size, trainable = False)
        
        else:
            #self.prior_y_init = prior_y
            self.prior_y = tf.Variable(tf.convert_to_tensor(prior_y, dtype = "float32"),
                                       trainable = False)
    
    def build(self, input_shape):
        self.input_size = input_shape[1]
        
        self.counts_memory = self.add_weight(name = "count_kernel",
                                             shape = (self.max_input_size + 1, 1),
                                             initializer = "ones",
                                             trainable = False)
        
        self.counts_memory[self.input_size].assign(self.input_size/self.max_input_size)
        #self.counts_memory[self.input_size].assign(0)
        
        #self.prior_y = self.add_weight(name = "prior_y", shape = (self.output_size + 1,),
        #                               initializer = Constant(self.prior_y_init),
        #                               trainable = False)
        
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name = "weigh_kernel",
                                      shape = (self.max_input_size + 1, self.output_size + 1),
                                      initializer = init.Categorical(self.prior_y, self.input_size),
                                      constraint = constr.AltOneNorm(self.input_size,
                                                                     self.prior_y, self.counts_memory,
                                                                     self.number_constraint_iterations),
                                      trainable = True)
        
        self.tau = self.add_weight(name = "tau", shape = (),
                                   initializer = Constant(self.tau_init),
                                   trainable = False)
    
    def call(self, h):
        
        @tf.custom_gradient
        def silent_normalization(kernel):
            def grad(upstream):
                
                downstream = upstream - k.sum(upstream * kernel, axis = 1, keepdims = True) / self.counts_memory[: self.input_size + 1]
                
                #downstream = ((self.input_size + self.counts_memory[self.input_size]) * self.prior_y - kernel) * (self.counts_memory[: self.input_size + 1] - kernel) / ((self.input_size + self.counts_memory[self.input_size]) * self.prior_y * self.counts_memory[: self.input_size + 1] - kernel**2) * upstream
                
                return downstream
            
            return kernel, grad
        
        kernel = silent_normalization(self.kernel[: self.input_size + 1])
        
        c = k.stop_gradient(k.max(h, axis = 1, keepdims = True))
        c = k.stop_gradient(k.maximum(c, self.tau))
        
        return c + k.log(k.dot(k.exp(h - c), kernel[: -1]) + k.exp(self.tau - c) * kernel[-1 :])
    
    # To support serialization
    def get_config(self):
        config = super(LogDenseExp, self).get_config()
        #config.update({"number_constraint_iterations" : self.number_constraint_iterations,
        #               "max_input_size" : self.max_input_size, "output_size" : self.output_size,
        #               "tau_init" : self.tau_init, "prior_y" : self.prior_y_init})
        config.update({"number_constraint_iterations" : self.number_constraint_iterations,
                       "max_input_size" : self.max_input_size, "output_size" : self.output_size,
                       "tau_init" : self.tau_init, "prior_y" : self.prior_y.value()})
        return config
    
    @classmethod
    def from_config(cls, config):
        prior_y_config = config.pop("prior_y")
        prior_y = tf.Variable(deserialize_keras_object(prior_y_config), trainable = False)
        
        return cls(**config, prior_y = prior_y)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size + 1)

### Layer that computes log(correlation(w, x)) using a convolution
### w and x are 2D with different shapes
class LogConvCor(Layer):

    def __init__(self, memory_shape, number_memories, memory_strides, w_mean, noise, **kwargs):
        self.memory_shape = memory_shape
        self.number_memories = number_memories
        self.memory_strides = memory_strides
        self.w_mean = w_mean
        self.noise = noise
        super(LogConvCor, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.K = k.cast(k.prod(self.memory_shape) * input_shape[-1], dtype = "float32")
        
        self.kernel = self.add_weight(name = "kernel",
                                      shape = self.memory_shape + (input_shape[-1], self.number_memories),
                                      initializer = init.ClusteredNormal(self.w_mean, self.noise, 1, axis = (0, 1, 2)),
                                      trainable = True)
        self.kernel.norm_order = "two"
        # Axes occupied by individual feature detectors
        self.kernel.axis = (0, 1, 2)
        super(LogConvCor, self).build(input_shape)
    
    def call(self, x):
        x_avg = k.pool2d(k.mean(x, axis = -1, keepdims = True),
                         self.memory_shape, self.memory_strides, pool_mode = "avg", padding = "same")
        
        x_var = self.K * (k.pool2d(k.mean(x**2, axis = -1, keepdims = True),
                                   self.memory_shape, self.memory_strides, pool_mode = "avg", padding = "same") - x_avg**2)
        
        x_is_blank = x_var == 0.
        
        h = k.log(tf.where(x_is_blank, 1., k.abs(k.conv2d(x, self.kernel, self.memory_strides, padding = "same")
                                                 - k.sum(self.kernel, axis = (0, 1, 2), keepdims = True) * x_avg)))
        
        h = h - (k.log(k.sum(self.kernel**2, axis = (0, 1, 2), keepdims = True))
                 + k.log(tf.where(x_is_blank, 1., x_var))) / 2
        
        return tf.where(x_is_blank, logzero, h)
    
    def get_config(self):
        base_config = super(LogConvCor, self).get_config()
        return {**base_config, "memory_shape" : self.memory_shape, "number_memories" : self.number_memories,
                "memory_strides" : self.memory_strides}

    def compute_output_shape(self, input_shape):
        return input_shape[: -1] + (self.number_memories,)

### Layer that computes log(dot(v, exp(beta*h)) + tau) using a convolution with 1 x 1 kernels
### Input and kernel are 2D with different shapes
class LogConvExp(Layer):

    def __init__(self, output_size, beta, tau, softening, **kwargs):
        self.output_size = output_size
        self.beta = beta
        self.tau = tau
        self.softening = softening
        super(LogConvExp, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name = "kernel",
                                      shape = (input_shape[-1], self.output_size),
                                      initializer = init.Categorical(self.softening, 1),
                                      trainable = True)
        self.kernel.norm_order = "one"
        
        super(LogConvExp, self).build(input_shape)
    
    def call(self, h):
        
        h = self.beta*h
        
        c = k.stop_gradient(k.max(h, axis = -1, keepdims = True))
        c = k.stop_gradient(k.maximum(c, self.tau))
        
        return k.log(k.dot(k.exp(h - c), self.kernel) + k.exp(self.tau - c))
    
    def get_config(self):
        base_config = super(LogConvExp, self).get_config()
        return {**base_config, "output_size" : self.output_size, "beta" : self.beta, "tau" : self.tau}

    def compute_output_shape(self, input_shape):
        return input_shape[: -1] + (self.output_size,)

### Calculate log(correlation(w, x)) when w and x are numpy arrays
def log_dense_cor(x, w):
    return np.log(np.abs(x @ w)) - (np.log(np.sum(w ** 2, axis = 0, keepdims = True))
                                    + np.log(np.sum(x ** 2, axis = 1, keepdims = True))) / 2
