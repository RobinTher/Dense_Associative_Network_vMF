from tensorflow.keras import backend as k
import tensorflow as tf
import numpy as np

from tensorflow.keras.initializers import Initializer
from tensorflow.keras.saving import deserialize_keras_object

import DAN_code.normalization as norm

from tensorflow.keras.utils import to_categorical

def sqrt_step(x):
    '''
    Helper function for the vMF sampling.
    '''
    x = k.abs(x)
    return tf.where(x < 1, x / ((x**2 + 1)**(1/2) + 1), 1 / ((1 + 1/x**2)**(1/2) + 1/x))

def tf_random_beta(number_samples, alpha, beta):
    '''
    Generate random samples from a Beta distribution.

    Parameters
    ----------
    number_samples : int
        The number of samples to generate.
    alpha : float
        The alpha parameter of the Beta distribution.
    beta : float
        The beta parameter of the Beta distribution.
    
    Returns
    -------
    tf.Tensor
        Samples from the Beta distribution.
    '''
    random_gamma_1 = tf.random.gamma(number_samples, alpha)
    random_gamma_2 = tf.random.gamma(number_samples, beta)
    
    random_beta = random_gamma_1 / (random_gamma_1 + random_gamma_2)
    return random_beta

def random_vmf_cos(sample_size, number_samples_sought, beta):
    '''
    Sample the cos of the angle between the mean direction and the sample direction
    in the von Mises-Fisher (vMF) distribution as in https://hal.science/hal-04004568v3/file/main.pdf.
    
    Parameters
    ----------
    sample_size : int
        The size of the sample space (dimension of the vMF distribution).
    number_samples_sought : int
        The number of samples to generate.
    beta : float or tf.Tensor
        The concentration parameter of the vMF distribution.
    
    Returns
    -------
    tf.Tensor
        Samples from the vMF distribution.
    '''
    beta = tf.cast(beta, dtype = "float32")
    rescaled_beta = 2*beta/(sample_size - 1)
    
    x = sqrt_step(rescaled_beta)
    c = beta * x + (sample_size - 1) * tf.math.log1p(-x**2)
    
    number_samples_found = 0
    samples_found = []
    while number_samples_found < number_samples_sought:
        number_samples_to_generate = np.minimum(number_samples_sought, int(3/2 * (number_samples_sought - number_samples_found)))
        
        random_beta = tf_random_beta([number_samples_to_generate], (sample_size - 1)/2, (sample_size - 1)/2)
        random_beta = (x + 1 - 2*random_beta) / (x + 1 - 2*x*random_beta)
        
        random_negxponential = k.log(tf.random.uniform([number_samples_to_generate]))
        
        accept_samples = beta * random_beta + (sample_size - 1) * tf.math.log1p(-x*random_beta) - c >= random_negxponential
        samples_found.append(random_beta[accept_samples])
        number_samples_found += len(samples_found[-1])
    
    return k.concatenate(samples_found)[: number_samples_sought]

def random_vmf(w, beta):
    '''
    Sample the von Mises-Fisher (vMF) distribution
    with the algorithm of https://hal.science/hal-04004568v3/file/main.pdf.

    Parameters
    ----------
    w : tf.Tensor
        Mean directions of the vMF distribution. Can use a different one for each sample.
    beta : float
        Concentration parameter of the vMF distribution.
    
    Returns
    -------
    w : tf.Tensor
        Samples from the vMF distribution with the same shape as the input w.
    '''
    z = tf.random.normal(w.shape)
    z /= norm.tensor_two_norm(z, axis = 0)
    
    z -= k.sum(z * w, axis = 0) * w
    z /= norm.tensor_two_norm(z, axis = 0)
    
    cos = random_vmf_cos(*w.shape, beta)
    sin = (1 - cos**2)**(1/2)
    
    w = cos * w + sin * z
    return w

def split_memories(model, mask, learning_rate):
    '''
    Duplicate the DAN memories and class weights indexed by the mask. Break the resulting
    permutation symmetry between these memories w and their duplicates w_dupli
    by updating them accordding to w_dupli <- w_dupli - learning_rate * eigvecs and
    w <- w + learning_rate * eigvecs, respectively, where eigvecs are trainable weights
    of DAN_code.layers.DenseCor that were learned during the splitting phase of training.
    Also update the number of hidden units of the DAN and the counts_memory of
    DAN_code.layers.LogDenseExp correspondingly.

    Parameters
    ----------
    model : DAN_code.models.DAN
        The DAN model to update.
    mask : tf.Tensor
        A boolean mask indicating which memories to duplicate.
    learning_rate : float
        The learning rate of the update.
    '''
    number_eigvals = k.sum(tf.cast(mask, dtype = "int32")).numpy()
    number_memories = model.get_DAN_layer(1).output_size
    memories = model.get_DAN_layer(1).kernel
    eigvecs = model.get_DAN_layer(1).eigvecs
    
    counts_memory = model.get_DAN_layer(2).counts_memory
    weighs = model.get_DAN_layer(2).kernel
    memories[:, number_memories : number_memories + number_eigvals].assign(tf.boolean_mask(memories - learning_rate * eigvecs, mask, axis = 1))
    memories.assign(tf.where(mask, memories + learning_rate * eigvecs, memories))
    memories[:, : number_memories + number_eigvals].assign(norm.tensor_normalize(memories[:, : number_memories + number_eigvals], norm.tensor_two_norm(memories[:, : number_memories + number_eigvals], axis = 0)))
    
    mask = k.concatenate([mask, tf.constant([False])])
    for var in [weighs, counts_memory]:
        var.assign(tf.where(mask[:, tf.newaxis], var / 2, var))
        var[number_memories + number_eigvals].assign(var[number_memories])
        var[number_memories : number_memories + number_eigvals].assign(tf.boolean_mask(var, mask, axis = 0))
        var[: number_memories + number_eigvals + 1].assign(tf.cast((number_memories + number_eigvals)/number_memories, dtype = "float32") * var[: number_memories + number_eigvals + 1])
        
    number_memories += number_eigvals
    
    eigvecs[:, : number_memories].assign(tf.random.normal((eigvecs.shape[0], number_memories)))
    eigvecs[:, : number_memories].assign(eigvecs[:, : number_memories] / norm.tensor_two_norm(eigvecs[:, : number_memories], axis = 0))
    
    model.get_DAN_layer(1).output_size = number_memories
    model.get_DAN_layer(1).kernel.assign(memories)
    model.get_DAN_layer(1).kernel.constraint.output_size = number_memories
    model.get_DAN_layer(1).eigvecs.assign(eigvecs)
    model.get_DAN_layer(1).eigvecs.constraint.output_size = number_memories
    
    model.get_DAN_layer(2).input_size = number_memories
    model.get_DAN_layer(2)._build_input_shape = (None, number_memories)
    model.get_DAN_layer(2).counts_memory.assign(counts_memory)
    model.get_DAN_layer(2).kernel.assign(weighs)
    model.get_DAN_layer(2).kernel.constraint.input_size = number_memories
    model.get_DAN_layer(2).kernel.constraint.counts_memory.assign(counts_memory)

def reinit_eigvecs(model, number_memories):
    eigvecs = model.get_DAN_layer(1).eigvecs
    
    eigvecs[:, : number_memories].assign(tf.random.normal((eigvecs.shape[0], number_memories)))
    eigvecs[:, : number_memories].assign(eigvecs[:, : number_memories] / norm.tensor_two_norm(eigvecs[:, : number_memories], axis = 0))

    eigvecs.constraint.output_size = number_memories
    
    model.get_DAN_layer(1).eigvecs.assign(eigvecs)
    model.get_DAN_layer(1).number_memories = number_memories
    model.get_DAN_layer(1).memories._trainable = False
    model.get_DAN_layer(1).weighs._trainable = False
    
    model.compile(optimizer = model.optimizer, loss = model.loss, metrics = [])

class RandomNormal(Initializer):

    def __init__(self, output_size):
        self.output_size = output_size
    
    def __call__(self, shape, dtype = None):
        memories = tf.Variable(tf.zeros(shape))
        
        memories[:, : self.output_size].assign(tf.random.normal((shape[0], self.output_size), stddev = 1/shape[0]**(1/2)))
        
        return memories
    
    # To support serialization
    def get_config(self):
        return {"output_size" : self.output_size}

class RandomSpherical(Initializer):
    '''
    Initialize the first output_size columns of a weight matrix uniformly at random
    on the unit hypersphere and set the rest to zero.
    In a DAN, this is used to initialize the first output_size memories
    and preallocate the rest for future use.

    Attributes
    ----------
    output_size : int
        The number of columns to initialize on the unit hypersphere.
    '''
    def __init__(self, output_size):
        self.output_size = output_size
    
    def __call__(self, shape, dtype = None):
        '''
        Call the initializer function.

        Parameters
        ----------
        shape : tuple
            The shape of the weight matrix to initialize.
        dtype : str, optional
            The data type of the weight matrix. Defaults to None.
        
        Returns
        -------
        memories : tf.Variable
            The initialized weight matrix.
        '''
        memories = tf.Variable(tf.zeros(shape))
        
        memories[:, : self.output_size].assign(tf.random.normal((shape[0], self.output_size)))
        memories[:, : self.output_size].assign(memories[:, : self.output_size] / norm.tensor_two_norm(memories[:, : self.output_size], axis = 0))
        
        return memories
    
    # To support serialization
    def get_config(self):
        return {"output_size" : self.output_size}

class Categorical(Initializer):
    '''
    Initialize the first input_size+1 rows of a weight matrix
    to prior_y, and rescale the last row by a factor of input_size/max_input_size,
    as explained in Appendix I of the paper. Initialize the rest to zero.
    max_input_size is obtained from the argument of the __call__ method.
    In a DAN, this is used to initialize the first input_size+1 class weights
    and preallocate the rest for future use.

    Attributes
    ----------
    prior_y : tf.Tensor
        A prior distribution over the classes.
    input_size : int
        The number of input activations.
    '''
    def __init__(self, prior_y, input_size):
        self.prior_y = prior_y
        self.input_size = input_size
    
    def __call__(self, shape, dtype = None):
        '''
        Call the initializer function.

        Parameters
        ----------
        shape : tuple
            The shape of the weight matrix to initialize.
        dtype : str, optional
            The data type of the weight matrix. Defaults to None.
        
        Returns
        -------
        weighs : tf.Variable
            The initialized weight matrix.
        '''
        weighs = tf.Variable(tf.zeros(shape))
        
        weighs[: self.input_size].assign(tf.ones((self.input_size, shape[1])) * self.prior_y)
        
        weighs[self.input_size].assign(self.input_size/shape[0] * self.prior_y)
        
        return weighs
    
    # To support serialization
    def get_config(self):
        return {"prior_y" : self.prior_y, "input_size" : self.input_size}
    
    @classmethod
    def from_config(cls, config):
        prior_y_config = config.pop("prior_y")
        prior_y = deserialize_keras_object(prior_y_config)
        
        return cls(**config, prior_y = prior_y)