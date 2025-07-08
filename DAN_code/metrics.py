from tensorflow.keras import backend as k
import tensorflow as tf

from tensorflow.keras.metrics import Metric

class RayleighQuotient(Metric):
    '''
    Monitor the progress of splitting steepest descent using the minimum
    of the Rayleigh quotients that it minimizes.

    Attributes
    ----------
    model : tf.keras.Model
        The model to monitor. It should have a DenseCor layer with an eigvecs attribute.
    metric : tf.keras.backend.variable
        The current value of the minimum Rayleigh quotient.
    batch : tf.keras.backend.variable
        The number of batches processed so far, used for averaging the metric.
    '''
    def __init__(self, model, **kwargs):
        super(RayleighQuotient, self).__init__(name = "rayleigh_quotient", **kwargs)
        
        self.model = model
        
        self.metric = self.add_weight(name = "metric", initializer = "zeros")
        self.batch = self.add_weight(name = "batch", initializer = "ones")
    
    def update_state(self, y_true, h_pred, sample_weight = None):
        '''
        Update the state of the metric. The Rayleigh quotient is a quadratic function,
        so we can compute it by multiplying the eigenvectors by the gradient
        of the loss with respect to them (see Appendix H of the paper).

        Parameters
        ----------
        y_true : tf.Tensor
            Placeholder true labels. Not used in this metric, but required by the Keras API.
        h_pred : tf.Tensor:
            The predicted logits.
        sample_weight : tf.Tensor, optional
            Not used in this metric, but required by the Keras API.
        '''
        eigvecs = self.model.get_DAN_layer(1).eigvecs
        
        scaled_eigvecs = k.gradients(self.model.compiled_loss(y_true, h_pred), eigvecs)[0]
        
        q = k.min(1/2 * k.sum(eigvecs * scaled_eigvecs, axis = 0))
        
        self.metric.assign_add((q - self.metric) / self.batch)
        self.batch.assign_add(1)
    
    def reset_state(self):
        '''
        Reset the state of the metric. This is called at the beginning of each epoch.
        '''
        self.metric.assign(0)
        self.batch.assign(1)
    
    def result(self):
        '''
        Return the current value of the metric.
        '''
        return self.metric
    
    # To support serialization
    def get_config(self):
        config = super(RayleighQuotient, self).get_config()
        config.update({"model" : self.model})
        return config