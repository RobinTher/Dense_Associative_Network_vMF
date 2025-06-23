from tensorflow.keras import backend as k
import tensorflow as tf
import numpy as np
#import numpy as np
#import tensorflow as tf
#import tensorflow.keras.backend as k

from tensorflow.keras.callbacks import Callback

import DAN_code.functions as func
import DAN_code.normalization as norm

class ParityTracker(Callback):
    
    def __init__(self, even_step):
        super(ParityTracker, self).__init__()
        self.even_step = even_step
    
    def on_batch_end(self, batch, logs = None):
        self.even_step = ~self.even_step

class BetaScheduler(Callback):
    
    def __init__(self, beta_final, slope, number_annealing_epochs):
        super(BetaScheduler, self).__init__()
        self.beta_final = beta_final
        self.slope = slope
        self.number_annealing_epochs = number_annealing_epochs
    
    def on_train_begin(self, logs = None):
        self.beta_init = self.model.get_DAN_layer(1).beta.value().numpy()
        self.number_features = self.model.get_DAN_layer(1).output_size
    
    def on_epoch_begin(self, epoch, logs = None):
        
        progress = (epoch + 1) / self.number_annealing_epochs
        
        if progress < 1:
            beta = (self.beta_final - self.beta_init) * progress**self.slope / (progress**self.slope + (1 - progress)**self.slope) + self.beta_init
        else:
            beta = self.beta_final
        
        self.model.get_DAN_layer(1).beta.assign(tf.cast(beta, dtype = "float32"))
        
        try:
            tau = func.log_gamma_ratio(beta, self.number_features)
            self.model.get_DAN_layer(2).tau.assign(tf.cast(tau, dtype = "float32"))
        except:
            pass

class AccuracyThresholdStopping(Callback):
    def __init__(self, accuracy_threshold):
        super(AccuracyThresholdStopping, self).__init__()
        self.accuracy_threshold = accuracy_threshold

    def on_epoch_end(self, epoch, logs = None): 
        val_accuracy = logs["val_accuracy"]
        if val_accuracy >= self.accuracy_threshold:
            self.model.stop_training = True

class ToggleMetric(Callback):
    def __init__(self, metric_name):
        super(ToggleMetric, self).__init__()
        self.metric_name = metric_name
    
    def on_test_begin(self, logs):
        for metric in self.model.metrics:
            if self.metric_name in metric.name:
                metric.active.assign(True)
    
    def on_test_end(self, logs):
        for metric in self.model.metrics:
            if self.metric_name in metric.name:
                metric.active.assign(False)

### Get weights during training.
class WeightEvolution(Callback):
    
    def __init__(self, beta, P, name, file_suffix):
        super(WeightEvolution, self).__init__()
        self.beta = beta
        self.P = P
        self.name = name
        self.file_suffix = file_suffix
        self.compress_labels = (file_suffix != "for_movies")
    
    #def on_batch_end(self, batch, logs = None):
    def on_epoch_end(self, batch, logs = None):
        w = self.model.get_DAN_layer(1).get_weights()[0][:, : self.P].T
        
        with open("./Data/Weights/%s_w_with_beta=%s_and_%s.npy" % (self.name, str(self.beta), self.file_suffix), "ab") as f:
            np.save(f, w)
        
        if len(self.model.layers) >= self.model.number_preproc_layers + 3:
            if self.compress_labels:
                #print("was here.")
                g = np.argmax(self.model.get_DAN_layer(2).get_weights()[0][: self.P], axis = -1)
            else:
                g = self.model.get_DAN_layer(2).get_weights()[0][: self.P].T
            
            with open("./Data/Weights/%s_g_with_beta=%s_and_%s.npy" % (self.name, str(self.beta), self.file_suffix), "ab") as f:
                np.save(f, g)

class AverageTransitionMatrix(Callback):
    
    def __init__(self, transition_matrix):
        super(AverageTransitionMatrix, self).__init__()
        self.transition_matrix = transition_matrix
        
        try:
            self.g_prev = np.argmax(self.model.get_DAN_layer(2).get_weights()[0], axis = 1)
        except:
            raise ValueError("Need a second layer of weights after the layer of memories in order to evaluate the average transition matrix.")
        
    
    def on_batch_end(self, batch, logs = None):
        g_next = np.argmax(self.model.get_DAN_layer(2).get_weights()[0], axis = 1)
        self.transition_matrix += np.histogram2d(self.g_prev, g_next, bins = self.model.output_shape[-1], density = False)[0]
        self.g_prev = g_next