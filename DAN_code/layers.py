from tensorflow.keras import backend as k
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant, Orthogonal
from tensorflow.keras.saving import deserialize_keras_object

import DAN_code.initializers as init
import DAN_code.constraints as constr
import DAN_code.normalization as norm
import DAN_code.functions as func

class Normalize(Layer):
    '''
    If normalize_online is True, normalize the input tensor along axis = 1
    by subtracting the mean and dividing by the two-norm.
    Otherwise, the input tensor is returned unchanged.

    Attributes
    ----------
    normalize_online : bool
        If True, normalize the input tensor online.
        Otherwise, it is assumed to be already normalized, so the the normalization is skipped.
    '''
    def __init__(self, normalize_online, **kwargs):
        super(Normalize, self).__init__(**kwargs)
        self.normalize_online = normalize_online
    
    def call(self, x):
        '''
        Call the layer on x, normalizing it if normalize_online is True.

        Parameters
        ----------
        x : tf.Tensor
            The input tensor.
        Returns
        -------
        x : tf.Tensor
            The normalized input tensor.
        '''
        if self.normalize_online:
            x = x - k.mean(x, axis = 1, keepdims = True)
            x = norm.tensor_normalize(x, norm.tensor_two_norm(x, axis = 1))
        
        return x
    
    # To support serialization
    def get_config(self):
        config = super(Normalize, self).get_config()
        config.update({"normalize_online" : self.normalize_online})
        return config

class DenseCor(Layer):
    '''
    Calculate the rescaled dot product beta * x @ w, between the data layer x of the DAN
    and its memories w, where the rescaling factor beta is the inverse temperature
    defined in the paper. The data layer is assumed to be normalized so that the output
    is the rescaled cosine similarity or Pearson correlation between the data and the memories.

    Attributes
    ----------
    output_size : int
        The number of hidden units that are connected to the layer at the start of training.
    max_output_size : int
        The maximum number of hidden units that can be connected to this layer.
        Weights for hidden units number output_size+1 to max_output_size are preallocated
        at initialization, but not immediately used in the network. Weights for these
        hidden units are built during training using splitting steepest descent.
    beta_init : float
        The initial value of the inverse temperature beta, used to scale the output.
    '''
    def __init__(self, output_size, max_output_size, beta_init, **kwargs):
        super(DenseCor, self).__init__(**kwargs)
        
        self.output_size = output_size

        self.max_output_size = max_output_size
        
        self.beta_init = beta_init
    
    def build(self, input_shape):
        '''
        Add weights to the layer. self.kernel contains the DAN memories,
        self.eigvecs contains the eigenvectors used for splitting steepest descent,
        and self.beta is the inverse temperature used to scale the output.

        Parameters
        ----------
        input_shape : tuple
            The shape of the input tensor, which is expected to be of the form
            (batch_size, input_size). input_size is the number of input activations.
        '''
        self.kernel = self.add_weight(name = "memory_kernel",
                                      shape = (input_shape[1], self.max_output_size),
                                      initializer = init.RandomSpherical(self.output_size),
                                      constraint = constr.UnitTwoNorm(self.output_size),
                                      trainable = True)
        
        self.eigvecs = self.add_weight(name = "eigvec_kernel",
                                       shape = (input_shape[1], self.max_output_size),
                                       initializer = init.RandomSpherical(self.output_size),
                                       constraint = constr.UnitTwoNorm(self.output_size),
                                       trainable = True)
        
        self.beta = self.add_weight(name = "beta", shape = (),
                                    initializer = Constant(self.beta_init),
                                    trainable = False)
    
    def call(self, x):
        '''
        Call the layer. The unveraged_rayleigh_quotient function, which evaluates
        the function F(phi ; theta, x) of the paper (see Appendix H),
        is included in the calculations in such a way that it does not contribute
        to the output activation, but only to the gradient with respect to self.eigvecs,
        which is used to learn self.eigvecs during the splitting phase of training.

        Parameters
        ----------
        x : tf.Tensor
            The data layer, which is assumed to be normalized.
        Returns
        -------
        tf.Tensor
            Dot product between the data layer and the DAN memories,
            scaled by the inverse temperature beta.
        '''
        @tf.custom_gradient
        def project_gradient(kernel):
            '''
            Project the gradient with respect to the input kernel onto its orthogonal complement,
            also known as the tangent space (see Appendix F).
            
            Parameters
            ----------
            kernel : tf.Tensor
                The kernel whose gradient is to be projected.
            Returns
            -------
            kernel : tf.Tensor
                The input kernel.
            grad : function
                A function that projects the gradient with respect to the input kernel
                onto its orthogonal complement.
            '''
            def grad(upstream):
                downstream = upstream - k.sum(upstream * kernel, axis = 0) * kernel
                return downstream
            
            return kernel, grad
        
        @tf.custom_gradient
        def stop_activation(activation):
            '''
            Multiply the input activation by zero and return a custom gradient function
            that returns the upstream gradient unchanged.

            Parameters
            ----------
            activation : tf.Tensor
                The input activation to be multiplied by zero.
            Returns
            -------
            activation : tf.Tensor
                The input activation multiplied by zero.
            grad : function
                A function that returns the upstream gradient unchanged.
            '''
            def grad(stream):
                return stream
            
            return 0 * activation, grad
        
        kernel = project_gradient(self.kernel[:, : self.output_size])
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
        '''
        Compute the output shape of the layer.
        Parameters
        ----------
        input_shape : tuple
            The shape of the input tensor, which is expected to be of the form
            (batch_size, input_size).
        Returns
        -------
        tuple
            The shape of the output tensor.
        '''
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
        
    def call(self, x):
        
        @tf.custom_gradient
        def project_gradient(kernel):
            def grad(upstream):
                reg = k.dot(kernel, tf.transpose(upstream))
                downstream = upstream - k.dot(kernel, (reg + tf.transpose(reg))/2)
                return downstream
            
            return kernel, grad
        
        kernel = project_gradient(self.kernel[: self.input_size, : self.output_size])
        
        return k.dot(x, kernel)
    
    # To support serialization
    def get_config(self):
        config = super(DenseOrth, self).get_config()
        config.update({"output_size" : self.output_size, "max_output_size" : self.max_output_size})
        return config
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)

class LogDenseExp(Layer):
    '''
    Calculate the shifted log dot product log(exp(h) @ g + Omega_N(beta)/Omega_N(0) * g_0)
    between the output h of the LogDenseCor layer h and the DAN class weights g and g_0.
    inverse temperature beta and Omega_N(kappa) are defined in the paper.

    Attributes
    ----------
    number_constraint_iterations : int
        The number of iterations to run the Sinkhorn-Knopp algorithm for the AltOneNorm constraint.
    input_size : int
        The number of input activations, which is inferred from the argument of the build method.
    max_input_size : int
        The maximum number of input activations, each corresponding to a hidden unit.
        Weights for hidden units number input_size+1 to max_input_size are preallocated
        at initialization, but not immediately used in the network. Weights for these
        hidden units are built during training using splitting steepest descent.
    output_size : int
        The number of classes in the DAN.
    tau_init : float
        The initial value of the tau parameter, calculated as tau = log(Omega_N(beta_init)/Omega_N(0))
        for a given initial inverse temperature beta_init in the models module.
    prior_y : tf.Variable, optional
        A prior distribution over the classes. If None, it is initialized to a uniform distribution.
        Defaults to None.
    '''
    def __init__(self, number_constraint_iterations, max_input_size,
                 output_size, tau_init, prior_y = None, **kwargs):
        super(LogDenseExp, self).__init__(**kwargs)
        
        self.number_constraint_iterations = number_constraint_iterations
        
        self.max_input_size = max_input_size

        self.output_size = output_size
        
        self.tau_init = tau_init
        
        if prior_y is None:
            self.prior_y = tf.Variable(tf.ones((output_size,)) / output_size, trainable = False)
        
        else:
            self.prior_y = tf.Variable(tf.convert_to_tensor(prior_y, dtype = "float32"),
                                       trainable = False)
    
    def build(self, input_shape):
        '''
        Add weights to the layer. self.kernel contains the DAN class weights g and g_0,
        in this order, self.counts_memory is used to normalize the rows of self.kernel,
        and self.tau is log(Omega_N(beta)/Omega_N(0)) for a given inverse temperature beta.

        Parameters
        ----------
        input_shape : tuple
            The shape of the input tensor, which is expected to be of the form
            (batch_size, input_size). input_size is the number of input activations.
        '''
        self.input_size = input_shape[1]
        
        self.counts_memory = self.add_weight(name = "count_kernel",
                                             shape = (self.max_input_size + 1, 1),
                                             initializer = "ones",
                                             trainable = False)
        
        self.counts_memory[self.input_size].assign(self.input_size/self.max_input_size)
        
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
        '''
        Call the layer. Use the logsumexp trick for numerical stability.

        Parameters
        ----------
        h : tf.Tensor
            The output of the LogDenseCor layer.
        Returns
        -------
        tf.Tensor
            The shifted log dot product between exp(h) and the DAN class weights.
        '''
        @tf.custom_gradient
        def project_gradient(kernel):
            '''
            Approximately project the gradient with respect to the input kernel onto
            its orthogonal complement, also known as the tangent space (see Appendix F).
            
            Parameters
            ----------
            kernel : tf.Tensor
                The kernel whose gradient is to be projected.
            Returns
            -------
            kernel : tf.Tensor
                The input kernel.
            grad : function
                A function that projects the gradient with respect to the input kernel
                onto its orthogonal complement.
            '''
            def grad(upstream):
                
                downstream = upstream - k.sum(upstream * kernel, axis = 1, keepdims = True) / self.counts_memory[: self.input_size + 1]
                
                return downstream
            
            return kernel, grad
        
        kernel = project_gradient(self.kernel[: self.input_size + 1])
        
        c = k.stop_gradient(k.max(h, axis = 1, keepdims = True))
        c = k.stop_gradient(k.maximum(c, self.tau))
        
        return c + k.log(k.dot(k.exp(h - c), kernel[: -1]) + k.exp(self.tau - c) * kernel[-1 :])
    
    # To support serialization
    def get_config(self):
        config = super(LogDenseExp, self).get_config()
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
        '''
        Compute the output shape of the layer.
        Parameters
        ----------
        input_shape : tuple
            The shape of the input tensor, which is expected to be of the form
            (batch_size, input_size).
        Returns
        -------
        tuple
            The shape of the output tensor.
        '''
        return (input_shape[0], self.output_size + 1)