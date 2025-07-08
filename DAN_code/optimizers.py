from tensorflow.keras import backend as k
import tensorflow as tf
import numpy as np

from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.utils import get_registered_name

import DAN_code.functions as func
import DAN_code.normalization as norm

class SMD(Optimizer):
    '''
    Optimizer designed for trainings DAN weights. Called SMD because it is close to
    stochastic mirror descent.

    Attributes
    ----------
    learning_rate : float or tf.Tensor
        The learning rate of the optimizer.
    momentum : float
        The momentum of the optimizer.
    smoothing : float
        A smoothing factor for computing running averages of the squared column-wise
        two-norms of the gradient of the loss with respect to the eigvec_kernel weight matrix
        of DAN_code.layer.DenseCor. These squared two-norms are also the squared Rayleigh
        quotients of the splitting matrices (see Appendix H). In the splitting phase of
        training, we divide the gradient with respect to eigvec_kernel by the square root
        of these squared Rayleigh quotients to accelerate convergence.
    name : str, optional
        The name of the optimizer. Defaults to "SMD".
    '''
    def __init__(self, learning_rate, momentum, smoothing, name = "SMD", **kwargs):
        
        super(SMD, self).__init__(name, **kwargs)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.momentum = momentum
        self.smoothing = smoothing
    
    def build(self, var_list):
        '''
        Initialize optimizer variables for each variable in var_list. self.given_names
        is initialized with the names of the variables in var_list. The variables named
        "weigh_kernel" and "eigvec_kernel", which belong to DAN_code.layer.DenseLogExp and
        DAN_code.layer.DenseCor, respectively, are optimized with custom update
        rules (see Appendices F and H). self.velocities contains exponentially moving
        averages of the gradients weighed by the learning rate and momentum.
        self.rayleigh_quotients_squared contains the running averages of the squared
        column-wise two-norms of the gradients with respect to eigvec_kernel.
        The entries of self.rayleigh_quotients_squared that correspond to variables
        other than eigvec_kernel are placeholders.

        Parameters
        ----------
        var_list : list of tf.Variable
            A list of variables to optimize for which to initialize optimizer variables.
        '''
        super(SMD, self).build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        
        self.given_names = []
        self.velocities = []
        self.rayleigh_quotients_squared = []
        for var in var_list:
            given_name = None
            for allowed_given_name in ["eigvec_kernel", "memory_kernel", "weigh_kernel"]:
                if allowed_given_name in var.name:
                    given_name = allowed_given_name
                    break
            
            self.given_names.append(given_name)
            
            self.velocities.append(self.add_variable_from_reference(var, "velocity"))
            
            if given_name == "eigvec_kernel":
                self.rayleigh_quotients_squared.append(self.add_variable_from_reference(var, "rayleigh_quotients_squared", shape = (var.shape[1],)))
            else:
                #self.rayleigh_quotients_squared.append(None)
                #self.rayleigh_quotients_squared.append(self.add_variable_from_reference(var, "placeholder", shape = ()))
                self.rayleigh_quotients_squared.append(self.add_variable_from_reference(var, "placeholder", shape = (var.shape[1],)))
    
    def update_step(self, grad, var):
        '''
        Update a trainable variable using an update rule that is specific to the
        variable's name, stored in self.given_names. The exponential update rule of weigh_kernel
        is similar to mirror descent with negative entropy divergence. The other update rules
        are closer to standard stochastic gradient descent, which is also a form of mirror descent.

        Parameters
        ----------
        grad : tf.Tensor
            The gradient of the loss with respect to the variable var.
        var : tf.Variable
            The variable to update.
        '''
        learning_rate = tf.cast(self.learning_rate, var.dtype)
        momentum = tf.cast(self.momentum, var.dtype)
        smoothing = tf.cast(self.smoothing, var.dtype)
        
        if isinstance(grad, tf.IndexedSlices):
            raise NotImplementedError("Sparse updates not implemented.")
        
        var_key = self._var_key(var)
        given_name = self.given_names[self._index_dict[var_key]]
        velocity = self.velocities[self._index_dict[var_key]]
        
        if given_name == "eigvec_kernel":
            rayleigh_quotients_squared = self.rayleigh_quotients_squared[self._index_dict[var_key]]
            
            rayleigh_quotients_squared.assign(smoothing * rayleigh_quotients_squared + (1 - smoothing) * k.sum(grad**2, axis = 0))
            
            grad = norm.tensor_normalize(grad - k.sum(var * grad, axis = 0) * var, rayleigh_quotients_squared**(1/2))
        
        velocity.assign(momentum * velocity - learning_rate * grad)
        
        if given_name == "weigh_kernel":
            var.assign(k.log(var) + learning_rate * velocity)
            var.assign(k.exp(var - k.max(var, axis = 1, keepdims = True)))

        else:
            var.assign(var + learning_rate * velocity)
    
    # To support serialization
    def get_config(self):
        config = super(SMD, self).get_config()
        config.update({"learning_rate" : self._serialize_hyperparameter(self.learning_rate),
                       "momentum" : self.momentum, "smoothing" : self.smoothing})
        return config