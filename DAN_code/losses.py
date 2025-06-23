from tensorflow.keras import backend as k
import tensorflow as tf
import numpy as np

from tensorflow.keras.losses import CategoricalCrossentropy, Loss

logeps = np.log(np.finfo(np.float32).eps)
logzero = -np.float32(151*np.log(2))

class SupervisedNegLogLikelihood(Loss):
    def __init__(self, softening):
        self.softening = softening
    
    def __call__(self, y_true, h_pred, sample_weight = None):
        super(SupervisedNegLogLikelihood, self).__init__()
        
        f_pred = k.sum(y_true * h_pred[:, : -1], axis = -1) + self.softening / (y_true.shape[1] + 1) * h_pred[:, -1]
        
        return -k.mean(f_pred)
    
    # To support serialization
    def get_config(self):
        return {"softening" : self.softening}

class UnsupervisedNegLogLikelihood(Loss):
    def __call__(self, y_true, h_pred, sample_weight = None):
        super(UnsupervisedNegLogLikelihood, self).__init__()
        
        y_pred = k.softmax(h_pred, axis = -1)
            
        f_pred = k.sum(y_pred * h_pred, axis = -1)
        
        return -k.mean(f_pred)

class NegLogLikelihood(Loss):
    def __init__(self, softening, supervised = True):
        self.softening = softening
        
        self.supervised = supervised
    
    def __call__(self, y_true, h_pred, sample_weight = None):
        super(NegLogLikelihood, self).__init__()
        if self.supervised:
            y_pred = y_true
            f_pred = k.sum(y_pred * h_pred[:, : -1], axis = -1) + self.softening / (y_pred.shape[1] + 1) * h_pred[:, -1]
            #f_pred = k.sum(y_pred * h_pred[:, : -1], axis = -1) + self.softening**2 / (y_pred.shape[1] + 1) * h_pred[:, -1]
        else:
            #y_pred = k.stop_gradient(k.softmax(h_pred, axis = -1))
            
            y_pred = k.softmax(h_pred, axis = -1)
            
            f_pred = k.sum(y_pred * h_pred, axis = -1)
        
        return -k.mean(f_pred)
    
    # To support serialization
    def get_config(self):
        return {"softening" : self.softening, "supervised" : self.supervised}
    
class NegCondLogLikelihood(NegLogLikelihood):
    def __init__(self, softening, supervised = True):
        super(NegCondLogLikelihood, self).__init__(softening, supervised)
    
    def __call__(self, y_true, h_pred, sample_weight = None):
        
        c = k.stop_gradient(k.max(h_pred, axis = -1, keepdims = True))
        h_pred = h_pred - c
        
        return super(NegCondLogLikelihood, self).__call__(y_true, h_pred - k.log(k.sum(k.exp(h_pred - c), axis = -1, keepdims = True)))
    
    # To support serialization
    def get_config(self):
        config = super(NegCondLogLikelihood, self).get_config()
        return config
        # return {"softening" : self.softening, "supervised" : self.supervised}

class Clustering():
    def __init__(self, beta, tau, gap):
        self.beta = beta
        self.tau = tau
        self.gap = gap
    
    def __call__(self, y_true, h_pred, sample_weight = None):
        
        h_pred = self.beta*h_pred
        c = k.stop_gradient(k.max(h_pred, axis = -1, keepdims = True))
        
        f_pred = c + k.log(k.sum(k.exp(h_pred - c), axis = -1, keepdims = True))
        
        c = k.stop_gradient(k.maximum(c, self.tau))
        
        f_pred = (1 - self.gap) * (f_pred - c) + self.gap * (self.tau - c) - k.log(k.exp(f_pred - c) + k.exp(self.tau - c))
        f_pred = tf.where(f_pred <= logzero, 0., f_pred)
        
        return -k.mean(f_pred)
    
    # To support serialization
    def get_config(self):
        return {"beta" : self.beta, "tau" : self.tau, "gap" : self.gap}