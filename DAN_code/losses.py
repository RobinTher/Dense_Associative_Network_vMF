from tensorflow.keras import backend as k
import tensorflow as tf
import numpy as np

from tensorflow.keras.losses import Loss

class SupervisedNegLogLikelihood(Loss):
    '''
    Effective loss used for training our model in a supervised manner.

    Parameters
    ----------
    softening : float
        The amount of label softening, also known as label smoothing.
    '''
    def __init__(self, softening):
        super(SupervisedNegLogLikelihood, self).__init__()
        self.softening = softening
    
    def __call__(self, y_true, h_pred, sample_weight = None):
        '''
        Evaluate the loss function.

        Parameters
        ----------
        y_true : tf.Tensor
            The true labels.
        h_pred : tf.Tensor
            The predicted logits.
        sample_weight : tf.Tensor, optional
            Not used in this loss, but required by the Keras API.
        
        Returns
        -------
        tf.Tensor
            The computed loss value.
        '''
        
        f_pred = k.sum(y_true * h_pred[:, : -1], axis = -1) + self.softening / (y_true.shape[1] + 1) * h_pred[:, -1]
        
        return -k.mean(f_pred)
    
    # To support serialization
    def get_config(self):
        return {"softening" : self.softening}

class UnsupervisedNegLogLikelihood(Loss):
    '''
    Effective loss used for training our model in an unsupervised manner.

    Parameters
    ----------
    softening : float
        The amount of label softening, also known as label smoothing.
    '''
    def __init__(self, softening):
        super(UnsupervisedNegLogLikelihood, self).__init__()
        self.softening = softening
    
    def __call__(self, y_true, h_pred, sample_weight = None):
        '''
        Evaluate the loss function.

        Parameters
        ----------
        y_true : tf.Tensor
            Placeholder true labels. Not used in this loss, but required by the Keras API.
        h_pred : tf.Tensor
            The predicted logits.
        sample_weight : tf.Tensor, optional
            The sample weights. Not used in this loss, but required by the Keras API.
        
        Returns
        -------
        tf.Tensor
            The computed loss value.
        '''
        y_pred = (1 - self.softening) * k.softmax(h_pred, axis = -1) + self.softening / h_pred.shape[-1]
            
        f_pred = k.sum(y_pred * h_pred, axis = -1)
        
        return -k.mean(f_pred)
    
    # To support serialization
    def get_config(self):
        return {"softening" : self.softening}