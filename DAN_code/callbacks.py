from tensorflow.keras import backend as k
import tensorflow as tf
import numpy as np

from tensorflow.keras.callbacks import Callback

import DAN_code.functions as func

class BetaEvolution(Callback):
    '''
    Callback the inverse temperature beta of the DAN during training.

    Attributes
    ----------
    name : str
        The name of the model, used to name the saved weights.
    '''
    def __init__(self, name):
        super(BetaEvolution, self).__init__()
        self.name = name
    
    def on_epoch_end(self, epoch, logs = None):
        '''
        Save the weights of the DAN at the end of each epoch.

        Parameters
        ----------
        epoch : int
            The index of the current epoch, starting from 0.
            Not used in this method, but kept for consistency with the callback interface.
        logs : dict, optional
            A dictionary containing logs of losses and metrics.
            Not used in this method, but can be useful for debugging.
        '''
        beta = np.squeeze(self.model.get_DAN_layer(1).beta.numpy())
        beta_reg = self.model.get_DAN_layer(1).beta_reg
        #print(beta)
        
        with open("./Data/Weights/%s_beta_with_beta_reg=%s.npy" % (self.name, str(beta_reg)), "ab") as f:
            np.save(f, beta)

class BetaScheduler(Callback):
    '''
    Cooling schedule that can be used for the inverse temperature beta of the DAN.
    The inverse temperature is annealed from its initial value to a final value
    using a power law. The slope of the power law
    and the number of annealing epochs can be adjusted.

    Attributes
    ----------
    beta_final : float
        The final value of beta.
    slope : float
        The slope of the power law used for annealing.
    number_annealing_epochs : int
        The number of epochs over which beta is annealed.
    beta_init : float
        The initial value of beta, set at the beginning of training.
    number_features : int
        The number of features in the input data, set at the beginning of training.
    '''
    def __init__(self, beta_final, slope, number_annealing_epochs):
        super(BetaScheduler, self).__init__()
        self.beta_final = beta_final
        self.slope = slope
        self.number_annealing_epochs = number_annealing_epochs
    
    def on_train_begin(self, logs = None):
        '''
        Initialize the beta parameter and the number of features at the beginning of training.
        This method retrieves the initial value of the inverse temperature beta
        and the number of features from the first DAN layer.
        It is called automatically by Keras when training starts.

        Parameters
        ----------
        logs : dict, optional
            A dictionary containing logs of losses and metrics.
            Not used in this method, but can be useful for debugging.
        '''
        self.beta_init = self.model.get_DAN_layer(1).beta.value().numpy()
        self.number_features = self.model.get_DAN_layer(1).input_shape[-1]
    
    def on_epoch_begin(self, epoch, logs = None):
        '''
        Update the inverse temperature beta at the beginning of each epoch
        using a power law based on the progress of the training.
        The progress is the ratio of the current epoch to the total number of annealing epochs.
        If the progress is less than 1, beta is updated using the power law formula.
        If the progress is 1 or more, beta is set to its final value.
        This method is called automatically at the start of each epoch during training.
        
        Parameters
        ----------
        epoch : int
            The index of the current epoch, starting from 0.
        logs : dict, optional
            A dictionary containing logs of losses and metrics.
            Not used in this method, but can be useful for debugging.
        '''
        progress = (epoch + 1) / self.number_annealing_epochs
        
        if progress < 1:
            beta = (self.beta_final - self.beta_init) * progress**self.slope / (progress**self.slope + (1 - progress)**self.slope) + self.beta_init
        else:
            beta = tf.constant([self.beta_final], dtype = "float32")
        
        self.model.get_DAN_layer(1).beta.assign(beta)

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
    '''
    Callback to save DAN weights during training.

    Attributes
    ----------
    beta_reg : float
        The regularization on beta, represented with the character varsigma (ς) in the paper. Used to name the saved weights.
    number_memories : int
        The number of DAN memories to save.
    name : str
        The name of the model, used to name the saved weights.
    file_suffix : str
        A suffix for the file name to distinguish the purpose of the saved weights.
    compress_class_weights : bool
        Set to True if and only the file_suffix is not "for_movies". Compress the class
        weights to their argmax if the weights are not saved to make movies. Such
        compressed class weights are also the classes of the corresponding memories.
    '''
    def __init__(self, beta_reg, number_memories, name, file_suffix):
        super(WeightEvolution, self).__init__()
        self.beta_reg = beta_reg
        self.number_memories = number_memories
        self.name = name
        self.file_suffix = file_suffix
        self.compress_class_weights = (file_suffix != "for_movies")
    
    def on_epoch_end(self, epoch, logs = None):
        '''
        Save the weights of the DAN at the end of each epoch.

        Parameters
        ----------
        epoch : int
            The index of the current epoch, starting from 0.
            Not used in this method, but kept for consistency with the callback interface.
        logs : dict, optional
            A dictionary containing logs of losses and metrics.
            Not used in this method, but can be useful for debugging.
        '''
        w = self.model.get_DAN_layer(1).get_weights()[0][:, : self.number_memories].T
        
        with open("./Data/Weights/%s_w_with_beta_reg=%s_and_%s.npy" % (self.name, str(self.beta_reg), self.file_suffix), "ab") as f:
            np.save(f, w)
        
        if len(self.model.layers) >= self.model.number_preproc_layers + 3:
            if self.compress_class_weights:
                g = np.argmax(self.model.get_DAN_layer(2).get_weights()[0][: self.number_memories], axis = -1)
            else:
                g = self.model.get_DAN_layer(2).get_weights()[0][: self.number_memories].T
            
            with open("./Data/Weights/%s_g_with_beta_reg=%s_and_%s.npy" % (self.name, str(self.beta_reg), self.file_suffix), "ab") as f:
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