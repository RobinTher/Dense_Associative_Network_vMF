from tensorflow.keras import backend as k
import tensorflow as tf

from tensorflow.keras.metrics import Metric

from DAN_code import functions as func

class RayleighQuotient(Metric):
    def __init__(self, model, **kwargs):
        super(RayleighQuotient, self).__init__(name = "rayleigh_quotient", **kwargs)
        
        self.model = model
        
        self.metric = self.add_weight(name = "metric", initializer = "zeros")
        self.batch = self.add_weight(name = "batch", initializer = "ones")
    
    def update_state(self, y_true, h_pred, sample_weight = None):
        eigvecs = self.model.get_DAN_layer(1).eigvecs
        
        scaled_eigvecs = k.gradients(self.model.compiled_loss(y_true, h_pred), eigvecs)[0]
        
        q = k.min(k.sum(eigvecs * scaled_eigvecs, axis = 0))
        
        self.metric.assign_add((q - self.metric) / self.batch)
        self.batch.assign_add(1)
    
    def reset_state(self):
        self.metric.assign(0)
        self.batch.assign(1)
    
    def result(self):
        return self.metric
    
    # To support serialization
    def get_config(self):
        config = super(RayleighQuotient, self).get_config()
        config.update({"model" : self.model})
        return config