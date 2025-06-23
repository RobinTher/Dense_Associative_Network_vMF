from tensorflow.keras import backend as k
import tensorflow as tf
import numpy as np

from tensorflow.keras.initializers import Initializer
from tensorflow.keras.saving import deserialize_keras_object

import DAN_code.normalization as norm

from tensorflow.keras.utils import to_categorical

def adjusted_divmod(number, base):
    return np.where(base == 0, (number, np.zeros_like(base)), np.divmod(number, base))

def sqrt_ramp(x):
    x = k.abs(x)
    return 1 / ((x**2 + 1)**(1/2) + x)

def sqrt_step(x):
    x = k.abs(x)
    return tf.where(x < 1, x / ((x**2 + 1)**(1/2) + 1), 1 / ((1 + 1/x**2)**(1/2) + 1/x))

# def sqrt_bump(x):
    # return 2 / ((x**2 + 1)**(1/2) + 1)

def tf_random_beta(number_samples, alpha, beta):
    random_gamma_1 = tf.random.gamma(number_samples, alpha)
    random_gamma_2 = tf.random.gamma(number_samples, beta)
    
    random_beta = random_gamma_1 / (random_gamma_1 + random_gamma_2)
    return random_beta

def random_vmf_cos(sample_size, number_samples_sought, beta):
    
    beta = tf.cast(beta, dtype = "float32")
    
    rescaled_beta = 2*beta/(sample_size - 1)
    
    b = sqrt_ramp(rescaled_beta)
    
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
    z = tf.random.normal(w.shape)
    z /= norm.tensor_two_norm(z, axis = 0)
    
    z -= k.sum(z * w, axis = 0) * w
    z /= norm.tensor_two_norm(z, axis = 0)
    
    cos = random_vmf_cos(*w.shape, beta)
    sin = (1 - cos**2)**(1/2)
    
    w = cos * w + sin * z
    return w

# def split_memories(number_memories, counts_memory, memories, weights, idx, eigvecs, learning_rate):
def split_memories(model, mask, learning_rate, number_preproc_layers = 1):
    number_eigvals = k.sum(tf.cast(mask, dtype = "int32")).numpy()
    
    number_memories = model.layers[number_preproc_layers + 1].output_size
    max_number_memories = model.layers[number_preproc_layers + 1].max_output_size
    
    memories = model.layers[number_preproc_layers + 1].kernel
    
    eigvecs = model.layers[number_preproc_layers + 1].eigvecs
    
    ###
    counts_memory = model.layers[number_preproc_layers + 2].counts_memory
    
    weighs = model.layers[number_preproc_layers + 2].kernel
    
    memories[:, number_memories : number_memories + number_eigvals].assign(tf.boolean_mask(memories - learning_rate * eigvecs, mask, axis = 1))
    
    memories.assign(tf.where(mask, memories + learning_rate * eigvecs, memories))
    
    memories[:, : number_memories + number_eigvals].assign(norm.tensor_normalize(memories[:, : number_memories + number_eigvals], norm.tensor_two_norm(memories[:, : number_memories + number_eigvals], axis = 0)))
    
    prev_ratio = (number_memories + 1) / (max_number_memories + 1)
    next_ratio = (number_memories + number_eigvals + 1) / (max_number_memories + 1)
    
    mask = k.concatenate([mask, tf.constant([False])])
    for var in [weighs, counts_memory]:
        
        var.assign(tf.where(mask[:, tf.newaxis], var / 2, var))
        
        var[number_memories + number_eigvals].assign(var[number_memories])
        
        var[number_memories : number_memories + number_eigvals].assign(tf.boolean_mask(var, mask, axis = 0))
        
        var[: number_memories + number_eigvals + 1].assign(tf.cast((number_memories + number_eigvals + next_ratio) / (number_memories + prev_ratio), dtype = "float32") * var[: number_memories + number_eigvals + 1])
        
    number_memories += number_eigvals
    
    eigvecs[:, : number_memories].assign(tf.random.normal((eigvecs.shape[0], number_memories)))
    eigvecs[:, : number_memories].assign(eigvecs[:, : number_memories] / norm.tensor_two_norm(eigvecs[:, : number_memories], axis = 0))
    
    model.layers[number_preproc_layers + 1].output_size = number_memories
    
    model.layers[number_preproc_layers + 1].kernel.assign(memories)
    model.layers[number_preproc_layers + 1].kernel.constraint.output_size = number_memories
    
    model.layers[number_preproc_layers + 1].eigvecs.assign(eigvecs)
    model.layers[number_preproc_layers + 1].eigvecs.constraint.output_size = number_memories
    
    ###
    model.layers[number_preproc_layers + 2].input_size = number_memories
    model.layers[number_preproc_layers + 2].counts_memory.assign(counts_memory)
    
    model.layers[number_preproc_layers + 2].kernel.assign(weighs)
    model.layers[number_preproc_layers + 2].kernel.constraint.input_size = number_memories
    model.layers[number_preproc_layers + 2].kernel.constraint.counts_memory.assign(counts_memory)

def reinit_eigvecs(model, number_memories, number_preproc_layers = 1):
    eigvecs = model.layers[number_preproc_layers + 1].eigvecs

    eigvecs[:, : number_memories].assign(tf.random.normal((eigvecs.shape[0], number_memories)))
    eigvecs[:, : number_memories].assign(eigvecs[:, : number_memories] / norm.tensor_two_norm(eigvecs[:, : number_memories], axis = 0))

    eigvecs.constraint.output_size = number_memories

    model.layers[number_preproc_layers + 1].eigvecs.assign(eigvecs)
    model.layers[number_preproc_layers + 1].number_memories = number_memories
    
    model.layers[number_preproc_layers + 1].memories._trainable = False
    model.layers[number_preproc_layers + 1].weighs._trainable = False
    
    model.compile(optimizer = model.optimizer, loss = model.loss, metrics = [])

class RandomSpherical(Initializer):
    
    def __init__(self, output_size):
        self.output_size = output_size
    
    def __call__(self, shape, dtype = None):
        memories = tf.Variable(tf.zeros(shape))
        
        memories[:, : self.output_size].assign(tf.random.normal((shape[0], self.output_size)))
        memories[:, : self.output_size].assign(memories[:, : self.output_size] / norm.tensor_two_norm(memories[:, : self.output_size], axis = 0))
        
        return memories
    
    # To support serialization
    def get_config(self):
        return {"output_size" : self.output_size}

class VMFMixture(Initializer):
    
    def __init__(self, beta, base_memories = None):
        
        self.beta = beta
        
        if base_memories is not None:
            base_memories = norm.array_normalize(base_memories, norm.array_two_norm(base_memories, axis = 0))
        
        self.base_memories = base_memories
        
    def __call__(self, shape, dtype = None):
        
        if self.base_memories is None:
            self.base_memories = tf.random.normal(shape)
            duplicated_memories = self.base_memories / norm.tensor_two_norm(self.base_memories, axis = 0)
        
        else:
            reps = tf.constant([1, shape[1] // self.base_memories.shape[1]])
            duplicated_memories = tf.tile(self.base_memories, reps)
            
            rest = shape[1] % self.base_memories.shape[1]
            noisy_memories = tf.random.normal((shape[0], rest))
            noisy_memories = noisy_memories / norm.tensor_two_norm(noisy_memories, axis = 0)
            duplicated_memories = tf.concat([noisy_memories, duplicated_memories], axis = 1)
        
        replicated_memories = random_vmf(duplicated_memories, self.beta)
        
        return replicated_memories
    
    # To support serialization
    def get_config(self):
        return {"number_memories" : self.number_memories, "beta" : self.beta, "base_memories" : self.base_memories}
    
    @classmethod
    def from_config(cls, config):
        base_memories_config = config.pop("base_memories")
        base_memories = deserialize_keras_object(base_memories_config)
        return cls(**config, base_memories = base_memories)

# class Categorical(Initializer):
    
    # def __init__(self, number_classes, softening, number_memories):
        # self.number_classes = number_classes
        # self.softening = softening
        # self.number_memories = number_memories
    
    # def __call__(self, shape, dtype = None):
        # total_number_memories = np.sum(self.number_memories)
        
        # if shape == (total_number_memories + 1, self.number_classes):
            # class_labels = np.arange(self.number_classes + 1).astype("int")
            # class_labels = np.repeat(class_labels, np.concatenate([self.number_memories, np.array([1])]))
            # class_labels = tf.one_hot(class_labels, self.number_classes)
            
            # kernel = (1 - self.softening) * class_labels + self.softening / (self.number_classes + 1)
            
            # kernel = kernel / ((1 - self.softening) * self.number_memories + self.softening * (total_number_memories + 1) / (self.number_classes + 1))
        
        # elif shape == (2, 1):
            # compl_labels = np.array([[0], [1]])
            
            # kernel = (1 - self.softening) * compl_labels + self.softening / (self.number_classes + 1)
            
            # kernel[0] = kernel[0] / ((1 - self.softening) / total_number_memories + (1 + 1/total_number_memories) / (self.number_classes + 1))
            # kernel[1] = kernel[1] / ((1 - self.softening) + self.softening * (total_number_memories + 1) / (self.number_classes + 1))
        
        # else:
            # raise ValueError("Shape mismatch. Valid shapes are (total_number_memories + 1, number_classes) for the class kernel and (2, 1) for the complement kernel.")
        
        # return kernel
    
    # To support serialization
    # def get_config(self):
        # return {"number_classes" : self.number_classes, "softening" : self.softening, "number_memories" : self.number_memories}

class Categorical(Initializer):
    
    def __init__(self, prior_y, input_size):
        self.prior_y = prior_y # tf.convert_to_tensor(prior_y, dtype = "float32")
        self.input_size = input_size
    
    def __call__(self, shape, dtype = None):
        weighs = tf.Variable(tf.zeros(shape))
        
        weighs[: self.input_size].assign(tf.ones((self.input_size, shape[1])) * self.prior_y)

        weighs[self.input_size].assign((self.input_size + 1) / (shape[0] + 1) * self.prior_y)
        #weighs[self.input_size].assign((self.input_size + 1) / (shape[0] + 1) * weighs[self.input_size])
        
        return weighs
    
    # To support serialization
    def get_config(self):
        return {"prior_y" : self.prior_y, "input_size" : self.input_size}
    
    @classmethod
    def from_config(cls, config):
        prior_y_config = config.pop("prior_y")
        prior_y = deserialize_keras_object(prior_y_config)
        
        return cls(**config, prior_y = prior_y)

class ExpCategorical(Initializer):
    
    def __init__(self, softening, scale):
        self.softening = softening
        self.scale = scale
    
    def __call__(self, shape, dtype = None):
        labels = np.arange(0, shape[1], shape[1]/shape[0]).astype("int")
        
        kernel = (1 - self.softening) * to_categorical(labels, shape[1]) + self.softening / shape[1]
        kernel = k.log(self.scale * kernel)
        return kernel
    
    # To support serialization
    def get_config(self):
        return {"softening" : self.softening, "scale" : self.scale}