import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as k

from tensorflow.keras.callbacks import Callback

import DAN_code.functions as func
import DAN_code.normalization as norm

class BetaScheduler(Callback):
    
    def __init__(self, beta_final, number_epochs, number_annealing_epochs, number_preproc_layers):
        super(BetaScheduler, self).__init__()
        self.beta_final = beta_final
        self.number_epochs = number_epochs
        self.number_preproc_layers = number_preproc_layers
        self.number_annealing_epochs = number_annealing_epochs
    
    def on_train_begin(self, logs = None):
        self.beta_init = self.model.layers[self.number_preproc_layers + 1].beta.value().numpy()
        self.number_features = self.model.layers[self.number_preproc_layers + 1].output_size
    
    def on_epoch_begin(self, epoch, logs = None):
        
        progress = (epoch + 1) / self.number_annealing_epochs
        
        # beta = (self.beta_final - self.beta_init) * progress + self.beta_init
        
        slope = 4
        if progress < 1:
            beta = (self.beta_final - self.beta_init) * progress**slope / (progress**slope + (1 - progress)**slope) + self.beta_init
        else:
            beta = self.beta_final
        
        # tf.print(beta)
        tau = func.log_gamma_ratio(beta, self.number_features)
        
        beta = tf.cast(beta, dtype = "float32")
        tau = tf.cast(tau, dtype = "float32")
        
        self.model.layers[self.number_preproc_layers + 1].beta.assign(beta) # .set_weights(memories_and_beta)
        self.model.layers[self.number_preproc_layers + 2].tau.assign(tau) # .set_weights(masses_and_tau)

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
    
    def __init__(self, beta, P, name, file_suffix, number_preproc_layers = 1):
    #def __init__(self, w_list, g_list, P, number_preproc_layers = 1):
        super(WeightEvolution, self).__init__()
        # w_list and g_list are lists used to collect w and g at each training batch.
        #self.w_list = w_list
        #self.g_list = g_list
        self.beta = beta
        self.P = P
        self.name = name
        self.file_suffix = file_suffix
        self.number_preproc_layers = number_preproc_layers
    
    #def on_batch_begin(self, batch, logs = None):
    def on_epoch_end(self, batch, logs = None):
        w = self.model.layers[self.number_preproc_layers + 1].get_weights()[0].T[: self.P]
        #self.w_list.append(w)
        with open("./Data/Weights/" + self.name + "_w_with_beta=" + str(self.beta)
                  + "_and_%s.npy" % self.file_suffix, "ab") as f:
            np.save(f, w)
        
        if len(self.model.layers) >= self.number_preproc_layers + 3:
            #g = self.model.layers[self.number_preproc_layers + 2].get_weights()[0][: self.P]#.T
            g = np.argmax(self.model.layers[self.number_preproc_layers + 2].get_weights()[0][: self.P], axis = -1)
            #self.g_list.append(g)
            with open("./Data/Weights/" + self.name + "_g_with_beta=" + str(self.beta)
                      + "_and_%s.npy" % self.file_suffix, "ab") as f:
                np.save(f, g)

class AverageTransitionMatrix(Callback):
    
    def __init__(self, transition_matrix, number_preproc_layers = 1):
        super(AverageTransitionMatrix, self).__init__()
        self.transition_matrix = transition_matrix
        self.number_preproc_layers = number_preproc_layers
        
        try:
            self.g_prev = np.argmax(self.model.layers[self.number_preproc_layers + 2].get_weights()[0], axis = 1)
        except:
            raise ValueError("Need a second layer of weights after the layer of memories in order to evaluate the average transition matrix.")
        
    
    def on_batch_end(self, batch, logs = None):
        g_next = np.argmax(self.model.layers[self.number_preproc_layers + 2].get_weights()[0], axis = 1)
        self.transition_matrix += np.histogram2d(self.g_prev, g_next, bins = self.model.output_shape[-1], density = False)[0]
        self.g_prev = g_next