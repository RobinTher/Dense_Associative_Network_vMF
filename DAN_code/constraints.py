from tensorflow.keras import backend as k
import tensorflow as tf
from tensorflow.keras.saving import deserialize_keras_object

from tensorflow.keras.constraints import Constraint
from tensorflow.keras.regularizers import Regularizer

import DAN_code.normalization as norm
import DAN_code.functions as func

class UnitTwoNorm(Constraint):
    '''
    Normalize the first output_size columns of a weight matrix w to have unit two-norm.
    The first output_size columns correspond to the output activations, while the other
    columns are preallocated but not used in the network.

    Attributes
    ----------
    output_size : int
        The number of columns to normalize.
    '''
    def __init__(self, output_size):
        self.output_size = output_size
        self.axis = 0
        #self.beta_sign = tf.constant(1., dtype = "float32")
    
    def __call__(self, w):
        '''
        Call the normalization function.

        Parameters
        ----------
        w : tf.Tensor
            The weight matrix to normalize.
        Returns
        -------
        w : tf.Tensor
            The normalized weight matrix.
        '''
        w_norm = norm.tensor_two_norm(w[:, : self.output_size], self.axis)
        w[:, : self.output_size].assign(norm.tensor_normalize(w[:, : self.output_size], w_norm))

        return w
    
    # To support serialization
    def get_config(self):
        return {"output_size" : self.output_size}

class AltOneNorm(Constraint):
    '''
    Normalize the first input_size + 1 rows of a weight matrix g to have one-norm equal
    to counts_memory[: input_size + 1] and the columns to have one-norm equal to
    (input_size + counts_memory[input_size]) * prior_y using the Sinkhorn-Knopp algorithm.
    The first input_size + 1 rows correspond to the input activations, while the other
    rows are preallocated but not used in the network. counts_memory is called G^gamma
    in the paper (see Appendix I), and prior_y is the prior distribution over the classes.
    
    Attributes
    ----------
    input_size : int
        The number of input activations.
    prior_y : tf.Tensor
        A prior distribution over the classes.
    counts_memory : tf.Tensor
        Used to normalize the rows of g.
    number_iterations : int
        The number of iterations to run the Sinkhorn-Knopp algorithm. Can be tuned.
    '''
    def __init__(self, input_size, prior_y, counts_memory, number_iterations):
        self.input_size = input_size
        self.max_input_size = counts_memory.shape[0]
        
        self.output_size = prior_y.shape[0]
        
        self.prior_y = prior_y
        self.counts_memory = counts_memory
        
        self.number_iterations = number_iterations
    
    def alt_normalize(self, g, not_all_normalized):
        '''
        One iteration of the Sinkhorn-Knopp algorithm.

        Parameters
        ----------
        g : tf.Tensor
            The weight matrix to normalize.
        not_all_normalized : tf.Tensor
            A boolean tensor indicating whether the normalization is complete.
            The normalization is complete when all rows still have their one-norm close to
            counts_memory[: input_size + 1] after the columns have been normalized.
        Returns
        -------
        g : tf.Tensor
            The normalized weight matrix.
        not_all_normalized : tf.Tensor
            A boolean tensor indicating whether the normalization is complete.
            The normalization is complete when all rows still have their one-norm close to
            counts_memory[: input_size + 1] after the columns have been normalized.
        '''
        g_norm = norm.tensor_one_norm(g, axis = 1)
        
        not_all_normalized = k.any(k.abs(g_norm / self.counts_memory[: self.input_size + 1] - 1) > (self.output_size - 1) * k.epsilon())
        #not_all_normalized = k.mean(k.abs(g_norm / self.counts_memory[: self.input_size + 1] - 1)) > (self.output_size - 1) * k.epsilon()
        
        g = self.counts_memory[: self.input_size + 1] * norm.tensor_normalize(g, g_norm)
        
        g_norm = norm.tensor_one_norm(g, axis = 0)
        g = (self.input_size + self.counts_memory[self.input_size]) * self.prior_y * norm.tensor_normalize(g, g_norm)
        
        return g, not_all_normalized
    
    def row_normalize(self, g):
        '''
        Currently not used.
        '''
        g_norm = norm.tensor_one_norm(g, axis = 1)
        g = self.counts_memory[: self.input_size + 1] * norm.tensor_normalize(g, g_norm)
        
        g_norm = norm.tensor_one_norm(g, axis = 0)
        g = norm.tensor_normalize(g, k.maximum(g_norm / ((self.input_size + self.counts_memory[self.input_size]) * self.prior_y), 1.))
        
        g_norm = norm.tensor_one_norm(g, axis = 1)
        g = norm.tensor_normalize(g, k.maximum(g_norm / self.counts_memory[: self.input_size + 1], 1.))
        
        g = g + ((self.input_size + self.counts_memory[self.input_size]) * self.prior_y - norm.tensor_one_norm(g, axis = 0)) * (self.counts_memory[: self.input_size + 1] - norm.tensor_one_norm(g, axis = 1)) / (self.input_size + self.counts_memory[self.input_size] - norm.tensor_one_norm(g, axis = None))
        
        return g
    
    def col_normalize(self, g):
        ''''
        Currently not used.
        '''
        g_norm = norm.tensor_one_norm(g, axis = 0)
        g = (self.input_size + self.counts_memory[self.input_size]) * self.prior_y * norm.tensor_normalize(g, g_norm)
        
        g_norm = norm.tensor_one_norm(g, axis = 1)
        g = norm.tensor_normalize(g, k.maximum(g_norm / self.counts_memory[: self.input_size + 1], 1.))
        
        g_norm = norm.tensor_one_norm(g, axis = 0)
        g = norm.tensor_normalize(g, k.maximum(g_norm / ((self.input_size + self.counts_memory[self.input_size]) * self.prior_y), 1.))
        
        g = g + ((self.input_size + self.counts_memory[self.input_size]) * self.prior_y - norm.tensor_one_norm(g, axis = 0)) * (self.counts_memory[: self.input_size + 1] - norm.tensor_one_norm(g, axis = 1)) / (self.input_size + self.counts_memory[self.input_size] - norm.tensor_one_norm(g, axis = None))
        
        return g
    
    def keep_looping(self, g, not_all_normalized):
        '''
        Check if the normalization is complete.
        '''
        return not_all_normalized
    
    def __call__(self, g):
        '''
        Call the Sinkhorn-Knopp algorithm for normalization.

        Parameters
        ----------
        g : tf.Tensor
            The weight matrix to normalize.
        Returns
        -------
        g : tf.Tensor
            The normalized weight matrix.
        '''
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