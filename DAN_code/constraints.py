from tensorflow.keras import backend as k
import tensorflow as tf
from tensorflow.keras.saving import deserialize_keras_object

from tensorflow.keras.constraints import Constraint

import DAN_code.normalization as norm

class BoundTwoNorm(Constraint):
    def __init__(self, w_bound, axis):
        self.w_bound = w_bound
        self.axis = axis
    
    def __call__(self, w):
        w_norm = norm.tensor_two_norm(w, self.axis)
        w_norm = tf.where(w_norm > self.w_bound, w_norm, 1)
        
        return norm.tensor_normalize(w, w_norm)
    
    # To support serialization
    def get_config(self):
        return {"w_bound" : self.w_bound, "axis" : self.axis}

class BoundOneNorm(Constraint):
    def __init__(self, v_bound, axis):
        self.v_bound = v_bound
        self.axis = axis
    
    def __call__(self, v):
        v_norm = norm.tensor_one_norm(v, self.axis)
        v_norm = tf.where(v_norm > self.v_bound, v_norm, 1)
        
        return norm.tensor_normalize(v, v_norm)
    
    # To support serialization
    def get_config(self):
        return {"v_bound" : self.v_bound, "axis" : self.axis}

class UnitTwoNorm(Constraint):
    def __init__(self, output_size, axis):
        self.output_size = output_size
        self.axis = axis
    
    def __call__(self, w):
        w_norm = norm.tensor_two_norm(w[:, : self.output_size], self.axis)
        w[:, : self.output_size].assign(norm.tensor_normalize(w[:, : self.output_size], w_norm))

        return w
    
    # To support serialization
    def get_config(self):
        return {"output_size" : self.output_size, "axis" : self.axis}

class UnitOneNorm(Constraint):
    def __init__(self, axis):
        self.axis = axis
    
    def __call__(self, g):
        g_norm = norm.tensor_one_norm(g, self.axis)
        
        return norm.tensor_normalize(g, g_norm)
    
    # To support serialization
    def get_config(self):
        return {"axis" : self.axis}

class AltOneNorm(Constraint):
    def __init__(self, input_size, prior_y, counts_memory, number_iterations):
        self.input_size = input_size
        self.max_input_size = counts_memory.shape[0]
        
        self.output_size = prior_y.shape[0]
        
        self.prior_y = prior_y
        self.counts_memory = counts_memory
        
        self.number_iterations = number_iterations
    
    def alt_normalize(self, g, not_all_normalized):
        
        g_norm = norm.tensor_one_norm(g, axis = 1)
        
        not_all_normalized = k.any(k.abs(g_norm / self.counts_memory[: self.input_size + 1] - 1) > (self.output_size - 1) * k.epsilon())
        #not_all_normalized = k.mean(k.abs(g_norm / self.counts_memory[: self.input_size + 1] - 1)) > (self.output_size - 1) * k.epsilon()
        
        g = self.counts_memory[: self.input_size + 1] * norm.tensor_normalize(g, g_norm)
        
        g_norm = norm.tensor_one_norm(g, axis = 0)
        g = (self.input_size + self.counts_memory[self.input_size]) * self.prior_y * norm.tensor_normalize(g, g_norm)
        
        return g, not_all_normalized
    
    def row_normalize(self, g):
        g_norm = norm.tensor_one_norm(g, axis = 1)
        g = self.counts_memory[: self.input_size + 1] * norm.tensor_normalize(g, g_norm)
        
        g_norm = norm.tensor_one_norm(g, axis = 0)
        g = norm.tensor_normalize(g, k.maximum(g_norm / ((self.input_size + self.counts_memory[self.input_size]) * self.prior_y), 1.))
        
        g_norm = norm.tensor_one_norm(g, axis = 1)
        g = norm.tensor_normalize(g, k.maximum(g_norm / self.counts_memory[: self.input_size + 1], 1.))
        
        g = g + ((self.input_size + self.counts_memory[self.input_size]) * self.prior_y - norm.tensor_one_norm(g, axis = 0)) * (self.counts_memory[: self.input_size + 1] - norm.tensor_one_norm(g, axis = 1)) / (self.input_size + self.counts_memory[self.input_size] - norm.tensor_one_norm(g, axis = None))
        
        return g
    
    def col_normalize(self, g):
        g_norm = norm.tensor_one_norm(g, axis = 0)
        g = (self.input_size + self.counts_memory[self.input_size]) * self.prior_y * norm.tensor_normalize(g, g_norm)
        
        g_norm = norm.tensor_one_norm(g, axis = 1)
        g = norm.tensor_normalize(g, k.maximum(g_norm / self.counts_memory[: self.input_size + 1], 1.))
        
        g_norm = norm.tensor_one_norm(g, axis = 0)
        g = norm.tensor_normalize(g, k.maximum(g_norm / ((self.input_size + self.counts_memory[self.input_size]) * self.prior_y), 1.))
        
        g = g + ((self.input_size + self.counts_memory[self.input_size]) * self.prior_y - norm.tensor_one_norm(g, axis = 0)) * (self.counts_memory[: self.input_size + 1] - norm.tensor_one_norm(g, axis = 1)) / (self.input_size + self.counts_memory[self.input_size] - norm.tensor_one_norm(g, axis = None))
        
        return g
    
    def keep_looping(self, g, not_all_normalized):
        return not_all_normalized
    
    def __call__(self, g):
        
        not_all_normalized = tf.constant(True)
        
        g[: self.input_size + 1].assign(tf.while_loop(self.keep_looping, self.alt_normalize, [g[: self.input_size + 1], not_all_normalized],
                                                      maximum_iterations = self.number_iterations)[0])
        
        #tf.print(norm.tensor_one_norm(g[: self.input_size + 1], axis = 1))
        #tf.print(norm.tensor_one_norm(g[: self.input_size + 1], axis = 0) / (self.input_size + self.counts_memory[self.input_size]))
        
        return g
    
    # To support serialization
    def get_config(self):
        return {"input_size" : self.input_size, "prior_y" : self.prior_y,
                "counts_memory" : self.counts_memory, "number_iterations" : self.number_iterations}
    
    @classmethod
    def from_config(cls, config):
        prior_y_config = config.pop("prior_y")
        prior_y = deserialize_keras_object(prior_y_config)
        
        counts_memory_config = config.pop("counts_memory")
        counts_memory = deserialize_keras_object(counts_memory_config)
        return cls(**config, prior_y = prior_y, counts_memory = counts_memory)

# exp(w) <- exp(w) / sum(exp(w)) => w = w - log(sum(exp(w)))

class UnitLogOneNorm(Constraint):
    def __init__(self, log_gap, axis):
        self.log_gap = log_gap
        self.axis = axis
    
    def __call__(self, v):
        v_norm = norm.tensor_log_one_norm(v, self.log_gap, self.axis)
        
        return norm.tensor_log_normalize(v, v_norm)
    
    # To support serialization
    def get_config(self):
        return {"log_gap" : self.log_gap, "axis" : self.axis}