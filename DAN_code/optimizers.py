from tensorflow.keras import backend as k
import tensorflow as tf
import numpy as np

from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.utils import get_registered_name

import DAN_code.functions as func
import DAN_code.normalization as norm

class NSD(Optimizer):
    def __init__(self, smoothing, learning_rate, momentum, softening, name = "NSD", **kwargs):
        super(NSD, self).__init__(name, **kwargs)
        self._set_hyper("smoothing", smoothing)
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("momentum", momentum)
        self._set_hyper("softening", softening)
    
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "average_grad")
            self.add_slot(var, "descent_dir")
    
    @tf.function
    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        
        smoothing = self._get_hyper("smoothing", var_dtype)
        learning_rate = self._decayed_lr(var_dtype)
        momentum = self._get_hyper("momentum", var_dtype)
        softening = self._get_hyper("softening", var_dtype)
        
        descent_dir = self.get_slot(var, "descent_dir")
        average_grad = self.get_slot(var, "average_grad")
        
        average_grad.assign(smoothing * average_grad + (1 - smoothing) * grad)
        grad = average_grad
        
        if "memory_kernel" in var.name:
            field = -norm.tensor_normalize(grad, norm.tensor_max_norm(grad, axis = 0))
            field = norm.tensor_normalize(field, norm.tensor_two_norm(field, axis = 0))
        
        elif "mass_kernel" in var.name:
            field = tf.where(k.min(grad, axis = 1, keepdims = True) == grad, 1., 0.)
            field = (1 - softening) * field + softening / var.shape[1]
        
        else:
            raise NotImplementedError("Divergence not implemented. Use 'memory_kernel' as the variable name for the Euclidean divergence and 'mass_kernel' for the Kullback-Leibler divergence.")
            
        descent_dir.assign(momentum * descent_dir + learning_rate * field)
        
        # var.assign_add(descent_dir)
        var.assign_add(momentum * descent_dir + learning_rate * field)
    
    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError
    
    # To support serialization
    def get_config(self):
        base_config = super(NSD, self).get_config()
        return {**base_config,
                "smoothing" : self._serialize_hyperparameter("smoothing"),
                "learning_rate" : self._serialize_hyperparameter("learning_rate"),
                "momentum" : self._serialize_hyperparameter("momentum"),
                "softening" : self._serialize_hyperparameter("softening")}

class SMD(Optimizer):
    def __init__(self, learning_rate, momentum, smoothing, name = "SMD", **kwargs):
        
        super(SMD, self).__init__(name, **kwargs)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.momentum = momentum
        self.smoothing = smoothing
    
    def build(self, var_list):
        super(SMD, self).build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        
        #self.var_given_name_dict = {}
        self.given_names = []
        self.velocities = []
        self.rayleigh_quotients_squared = []
        for var in var_list:
            given_name = None
            for allowed_given_name in ["eigvec_kernel", "memory_kernel", "weigh_kernel"]:
                if allowed_given_name in var.name:
                    #self.var_given_name_dict.update({var.name : allowed_var_given_name})
                    given_name = allowed_given_name
                    break
            
            self.given_names.append(given_name)
            
            self.velocities.append(self.add_variable_from_reference(var, "velocity"))
            
            # if "eigvec_kernel" in var.name:
            if given_name == "eigvec_kernel":
                self.rayleigh_quotients_squared.append(self.add_variable_from_reference(var, "rayleigh_quotients_squared", shape = (var.shape[1],)))
            else:
                #self.rayleigh_quotients_squared.append(None)
                #self.rayleigh_quotients_squared.append(self.add_variable_from_reference(var, "placeholder", shape = ()))
                self.rayleigh_quotients_squared.append(self.add_variable_from_reference(var, "placeholder", shape = (var.shape[1],)))
    
    def update_step(self, grad, var):
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
        
        # noise = self.generator.normal(var.shape)
        # velocity.assign(momentum * velocity + ((1 - momentum**2)/self.dataset_size)**(1/2) * noise - learning_rate * grad)
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