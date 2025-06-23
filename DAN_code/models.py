from tensorflow.keras import backend as k
import tensorflow as tf
import numpy as np

from tensorflow.keras import Model, Sequential
from tensorflow.keras.models import load_model
# from tensorflow.keras.models import functional
from tensorflow.keras.layers import Input, Dense, BatchNormalization, deserialize, Layer
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.callbacks import TerminateOnNaN, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import Metric
from tensorflow.keras.saving import serialize_keras_object, deserialize_keras_object
from tensorflow.keras.utils import register_keras_serializable

import DAN_code.functions as func
from DAN_code import layers
from DAN_code import losses
import DAN_code.normalization as norm
from DAN_code.optimizers import SMD, NSD
from DAN_code import callbacks
from DAN_code import metrics

import matplotlib.pyplot as plt
import DAN_code.result_plotting as res_plot

from DAN_code.result_plotting import animate

from DAN_code.serialize_custom_objects import collect_custom_objects

from datetime import datetime
from packaging import version

class DAN(Sequential):
    def __init__(self, *args, number_preproc_layers = 0, **kwargs):
        super(DAN, self).__init__(*args, **kwargs)
       
        self.number_preproc_layers = number_preproc_layers
        
    def get_DAN_layer(self, index):
        return self.layers[self.number_preproc_layers + index]
    
    def get_config(self):
        config = super(DAN, self).get_config()
        
        config.update({"number_preproc_layers" : self.number_preproc_layers})
        
        return config

def init_vanilla_net(softening, number_features, number_units, number_classes,
                     learning_rate, momentum, regularization, normalize_online = False):
    
    inputs = Input(shape = (number_features,))
    normalized_inputs = layers.Normalize(normalize_online)(inputs)
    outputs = Dense(number_units, activation = "softplus",
                    kernel_regularizer = L1L2(l1 = regularization, l2 = regularization))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Dense(number_classes + 1, activation = "softmax",
                    kernel_regularizer = L1L2(l1 = regularization, l2 = regularization))(outputs)
    
    model = Model(inputs = inputs, outputs = outputs)
    
    optimizer = SGD(learning_rate, momentum, nesterov = True)
    
    model.compile(optimizer, loss = losses.NegCondLogLikelihood(softening, supervised = True),
                  metrics = ["accuracy"])
    
    return model

### Initialize a new DAN model.
def init_normal_DAN(loss, beta_init, softening, number_features, number_memories, max_number_memories,
                    number_classes, number_constraint_iterations, learning_rate, momentum,
                    prior_y = None, normalize_online = False):
    
    # Set up the network structure
    inputs = Input(shape = (number_features,))
    normalized_inputs = layers.Normalize(normalize_online)(inputs)
    
    outputs = layers.DenseNormalDot(number_memories, max_number_memories, beta_init)(normalized_inputs)
    
    outputs = layers.LogDenseNormalExp(number_constraint_iterations, max_number_memories,
                                       number_classes, prior_y)(outputs)
    
    model = DAN(inputs = inputs, outputs = outputs)
    
    optimizer = SMD(learning_rate, momentum, momentum)
    
    if loss == "supervised":
        model.compile(optimizer, loss = losses.NegLogLikelihood(softening, supervised = True),
                      metrics = ["accuracy"])
    elif loss == "unsupervised":
        model.compile(optimizer, loss = losses.NegLogLikelihood(softening, supervised = False),
                      metrics = [])
    else:
        raise ValueError("Loss type not recognized. Supported values are 'supervised' and 'unsupervised'.")

    return model

### Initialize a new DAN model.
def init_DAN(loss, beta_init, softening, number_features, number_memories, max_number_memories,
             number_classes, number_constraint_iterations, learning_rate, momentum,
             prior_y = None, normalize_online = False):
    
    tau_init = func.log_gamma_ratio(beta_init, number_features)
    
    # Set up the network structure
    model = DAN(number_preproc_layers = 0)
    
    model.add(Input(shape = (number_features,)))
    model.add(layers.Normalize(normalize_online))
    model.add(layers.DenseCor(number_memories, max_number_memories, beta_init))
    model.add(layers.LogDenseExp(number_constraint_iterations, max_number_memories,
                                 number_classes, tau_init, prior_y))
    
    optimizer = SMD(learning_rate, momentum, momentum)
    
    if loss == "supervised":
        model.compile(optimizer, loss = losses.SupervisedNegLogLikelihood(softening),
                      metrics = ["accuracy"])
        #model.compile(optimizer, loss = losses.NegLogLikelihood(softening, supervised = True),
        #              metrics = ["accuracy"])
    elif loss == "unsupervised":
        model.compile(optimizer, loss = losses.UnsupervisedNegLogLikelihood(softening),
                      metrics = [])
        #model.compile(optimizer, loss = losses.NegLogLikelihood(softening, supervised = False),
        #              metrics = [])
    else:
        raise ValueError("Loss type not recognized. Supported values are 'supervised' and 'unsupervised'.")
    
    return model

def calc_split_mask(x_train, y_train, model, max_number_eigvals, max_eigval, batch_size, adjust = True):
    
    number_memories = model.get_DAN_layer(1).output_size
    max_number_memories = model.get_DAN_layer(1).max_output_size
    eigvecs = model.get_DAN_layer(1).eigvecs
    
    compiled_acc_scaled_eigvecs = tf.function(acc_scaled_eigvecs)
    
    data_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    data_train = data_train.batch(batch_size)
    
    scaled_eigvecs = tf.Variable(tf.zeros_like(eigvecs))
    running_weights = tf.Variable(tf.constant(0, dtype = "float32"))
    for x_batch, y_batch in data_train:
        compiled_acc_scaled_eigvecs(x_batch, y_batch, running_weights, eigvecs, scaled_eigvecs, model)
    
    if adjust:
        eigvecs[:, : number_memories].assign(norm.tensor_normalize(scaled_eigvecs[:, : number_memories], norm.tensor_two_norm(scaled_eigvecs[:, : number_memories], axis = 0)))
        
        scaled_eigvecs.assign(tf.zeros_like(eigvecs))
        running_weights.assign(0)
        for x_batch, y_batch in data_train:
            compiled_acc_scaled_eigvecs(x_batch, y_batch, running_weights, eigvecs, scaled_eigvecs, model)
    
    eigvals = k.sum(eigvecs * scaled_eigvecs, axis = 0)
    
    # print(eigvals)
    # print([rayleigh_quotients for rayleigh_quotients in model.optimizer.rayleigh_quotients_squared if rayleigh_quotients is not None][0])
    
    # u_plot = eigvecs[:, : 25].numpy()
        
    # res_plot.plot_images(u_plot.T)
    
    # u_scaled_plot = scaled_eigvecs[:, : 25].numpy()
        
    # res_plot.plot_images(u_scaled_plot.T)
    
    # r_plot = u_plot - 1/eigvals[: 25].numpy() * u_scaled_plot
    # res_plot.plot_images(r_plot.T)
    
    max_number_eigvals = np.min([max_number_eigvals, number_memories, max_number_memories - number_memories])
    mask = eigvals < k.minimum(func.kth_min(eigvals, k = max_number_eigvals + 1), max_eigval)
    
    return mask

def acc_scaled_eigvecs(x_batch, y_batch, running_weights, eigvecs, scaled_eigvecs, model):
    running_weights.assign_add(x_batch.shape[0])
    with tf.GradientTape() as tape:
        tape.watch(eigvecs)
        h_pred = model(x_batch)
        
        L_batch = model.compiled_loss(y_batch, h_pred)
        
        # tf.print(tf.where(k.sum(tape.gradient(L_batch, eigvecs)**2, axis = 0)))
        scaled_eigvecs.assign_add(x_batch.shape[0]/running_weights * (tape.gradient(L_batch, eigvecs) - scaled_eigvecs))

    return L_batch


def init_cluster_CAN(beta, tau, input_shape, memory_shape,
                     number_memories, memory_strides,
                     smoothing, learning_rate, momentum, softening):
    
    # Set up the network structure
    inputs = Input(shape = input_shape)
    outputs = layers.LogConvCor(memory_shape, number_memories, memory_strides, 0, np.inf)(inputs)
    model = Model(inputs = inputs, outputs = outputs)
    
    model.beta = beta
    model.tau = tau
    
    # Set up the optimizer
    nsd = NSD(smoothing, learning_rate, momentum, softening)
    
    model.compile(nsd, loss = losses.Clustering(beta, tau), metrics = [])
    
    return model

### Initialize a new CAN model.
def init_CAN(loss, beta, tau, input_shape, memory_shape,
             number_memories, memory_strides, number_classes,
             smoothing, learning_rate, momentum, softening, noise, w_mean):
    
    if loss == "supervised":
        v_softening = 1
    elif loss == "unsupervised":
        v_softening = softening
    else:
        raise NotImplementedError("Loss not recognized.")
    
    # Set up the network structure
    inputs = Input(shape = input_shape)
    outputs = layers.LogConvCor(memory_shape, number_memories, memory_strides, w_mean, noise)(inputs)
    outputs = layers.LogConvExp(number_classes, beta, tau, v_softening)(outputs)
    model = Model(inputs = inputs, outputs = outputs)
    
    model.beta = beta
    model.tau = tau
    
    # Set up the optimizer
    nsd = NSD(smoothing, learning_rate, momentum, softening)
    
    if loss == "supervised":
        model.compile(nsd, loss = losses.Supervised(softening), metrics = ["accuracy"])
    elif loss == "unsupervised":
        model.compile(nsd, loss = losses.Unsupervised(number_classes, softening), metrics = [])
    else:
        raise NotImplementedError("Loss not recognized.")
    
    return model

#def compile_model(model, softening, learning_rate, momentum):
#    # Set up the optimizer
#    optimizer = SMD(learning_rate, momentum)
#    # optimizer = SGD(learning_rate, momentum)
#    # optimizer = NSD(smoothing, learning_rate, momentum, softening)
#    
#    model.compile(optimizer, loss = losses.NegLogLikelihood(number_classes, softening, supervised = False), metrics = [])
#    
#    return model
    
def compile_memorization_phase(model):
    model.get_DAN_layer(1).kernel._trainable = True
    model.get_DAN_layer(1).eigvecs._trainable = False
    model.get_DAN_layer(2).kernel._trainable = True
    
    model.compile(optimizer = model.optimizer, loss = model.loss, metrics = ["accuracy"])

def compile_splitting_phase(model):
    model.get_DAN_layer(1).kernel._trainable = False
    model.get_DAN_layer(1).eigvecs._trainable = True
    model.get_DAN_layer(2).kernel._trainable = False
    
    model.compile(optimizer = model.optimizer, loss = model.loss,
                  metrics = [metrics.RayleighQuotient(model)])

### Train a vanilla neural net
def train_vanilla_net(x_train, y_train, model, number_epochs, batch_size, verbose = False):
    callback_list = []
    if verbose == True:
        callback_list.append(TerminateOnNaN())
        
    # Train model
    model.fit(x_train, y_train, epochs = number_epochs, batch_size = batch_size,
              verbose = verbose, callbacks = callback_list, validation_split = 0.1)
    
    return model

### Train a DAN.
def train_DAN(x_train, y_train, model, number_epochs, number_annealing_epochs,
              batch_size, beta_final = None, slope = 1, patience = 0,
              training_phase = "memorization", record = None, verbose = False,
              name = "DAN", validation_split = 0.1):
    
    try:
        beta_init = model.get_DAN_layer(1).beta.value().numpy()
    except AttributeError:
        beta_init = model.get_DAN_layer(1).beta_init
    
    number_memories = model.get_DAN_layer(1).output_size
    
    if beta_final is None:
        beta_final = beta_init
    
    if beta_final == beta_init:
        number_annealing_epochs = 0
    
    if training_phase == "memorization":
        compile_memorization_phase(model)
        
        if validation_split == 0:
            monitored_quantity = "loss"
        else:
            monitored_quantity = "val_loss"
        
    elif training_phase == "splitting":
        compile_splitting_phase(model)
        
        if validation_split == 0:
            monitored_quantity = "rayleigh_quotient"
        else:
            monitored_quantity = "val_rayleigh_quotient"
        
        record = None
        
    else:
        raise ValueError("Training phase not supported. Expected 'memorization' or 'splitting'.")
    
    callback_list = []
    
    if number_annealing_epochs != 0:
        callback_list.append(callbacks.BetaScheduler(beta_final, slope,
                                                     number_annealing_epochs))
    
    if patience is not None:
        callback_list.append(EarlyStopping(monitor = monitored_quantity, patience = patience, mode = "min", start_from_epoch = number_annealing_epochs))
    
    if verbose == True:
        callback_list.append(TerminateOnNaN())
    
    if record == None:
        pass
    elif record == "movies":
        P = 25
        callback_list.append([callbacks.WeightEvolution(beta_final, P, name, "for_movies")])
    elif record == "weights_with_splitting" or "weights_without_splitting":
        # 2000 is a safety upper bound on the number of memories to save.
        P = np.minimum(2000, number_memories)
        callback_list.append([callbacks.WeightEvolution(beta_final, P, name, record[8 :])])
    elif record == "transitions":
        transition_matrix = np.zeros((model.output_shape[-1], model.output_shape[-1]))
        callback_list.append([callbacks.AverageTransitionMatrix(transition_matrix)])
    else:
        raise ValueError("Record type not supported. Expected 'movies', 'weights_with_splitting', 'weights_without_splitting' or 'transitions'.")
    
    # Train model
    model.fit(x_train, y_train, epochs = number_epochs, batch_size = batch_size,
              verbose = verbose, callbacks = callback_list, validation_split = validation_split)
    
    beta = beta_final
    
    if record == "movies":
        w_list = load_contents("./Data/Weights/%s_w_with_beta=%s_and_for_movies.npy" % (name, str(beta_final)))
        g_list = load_contents("./Data/Weights/%s_g_with_beta=%s_and_for_movies.npy" % (name, str(beta_final)))
        
        animate(w_list, g_list, model)
    
    elif record == "transitions":
        with open("./Data/%s_transition_matrix_with_beta=%s_and_%s_memories.npy" % (name, str(beta), str(number_memories)), "wb") as f:
            np.fill_diagonal(transition_matrix, np.nan)
            transition_matrix = transition_matrix.T
            np.save(f, transition_matrix)
    
    return model

### Save a trained vanilla neural network.
def save_vanilla_net(model, number_units, name = "vanilla_net"):
    model.save("./Data/Nets/%s_with_%s_hidden_units.keras" % (name, str(number_units)))

### Load a trained vanilla neural network.
def load_vanilla_net(number_units, name = "vanilla_net"):
    return load_model("./Data/Nets/%s_with_%s_hidden_units.keras" % (name, str(number_units)))

### Save a trained DAN.
def save_DAN(model, beta, number_memories, number_splits, name = "DAN"):
    
    model.save("./Data/Nets/%s_with_beta=%s_and_%s_memories_for_%s_splits.keras"
               % (name, str(beta), str(number_memories), str(number_splits)))

### load a trained DAN.
def load_DAN(beta, number_memories, number_splits, custom_objects, name = "DAN"):
    
    model = load_model("./Data/Nets/%s_with_beta=%s_and_%s_memories_for_%s_splits.keras"
                       % (name, str(beta), str(number_memories), str(number_splits)),
                       custom_objects = custom_objects)
    return model

def load_contents(filename):
    contents = []
    
    with open(filename, "rb") as f:
        while True:
            try:
                contents.append(np.load(f))
            except ValueError:
                break
    
    return contents