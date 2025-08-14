from tensorflow.keras import backend as k
import tensorflow as tf
import numpy as np

from tensorflow.keras import Model, Sequential
from tensorflow.keras.models import load_model
# from tensorflow.keras.models import functional
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.callbacks import TerminateOnNaN, LearningRateScheduler, EarlyStopping
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import InverseTimeDecay

import DAN_code.functions as func
from DAN_code import layers
from DAN_code import losses
import DAN_code.normalization as norm
from DAN_code.optimizers import SMD
from DAN_code import callbacks
from DAN_code import metrics

from DAN_code.result_plotting import animate

from DAN_code.serialize_custom_objects import collect_custom_objects

class DAN(Sequential):
    '''
    Dense Associative Network (DAN) model class. We allow for a variable number
    of preprocessing layers before the DAN layers implemented in DAN_code.layers.py.
    Called DAM in the paper, where there are no preprocessing layers.

    Attributes
    ----------
    number_preproc_layers : int
        Number of preprocessing layers in the model.
    '''
    def __init__(self, *args, number_preproc_layers = 0, **kwargs):
        super(DAN, self).__init__(*args, **kwargs)
       
        self.number_preproc_layers = number_preproc_layers
        
    def get_DAN_layer(self, index):
        '''
        Get the DAN layer at the specified index, bypassing the preprocessing layers.

        Parameters
        ----------
        index : int
            The index of the DAN layer to retrieve. In our code, the zeroth index is
            reserved for normalization with DAN_code.layers.Normalize, the first index is
            for the DAN_code.layers.DenseCor, and the second index is for DAN_code.layers.LogDenseExp.
            It could be possible to start with a different normalization layer. On the other hand,
            bypassing the normalization layer and starting with DAN_code.layers.DenseCor
            would make the model incompatible with the current code, as it frequently uses
            get_DAN_layer(1) to access the DenseCor layer and get_DAN_layer(2) to access the
            LogDenseExp layer. For no normalization, set the normalize_online argument of
            the Normalize layer to False instead.
        Returns
        -------
        Layer
            The DAN layer at the specified index, bypassing the preprocessing layers.
        '''
        return self.layers[self.number_preproc_layers + index]
    
    def get_config(self):
        config = super(DAN, self).get_config()
        
        config.update({"number_preproc_layers" : self.number_preproc_layers})
        
        return config

# Collect custom objects defined in this module
custom_objects = collect_custom_objects({"DAN" : DAN})

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

def init_DAN(loss, beta_init, beta_reg, softening, number_features, number_memories,
             max_number_memories, number_classes, number_constraint_iterations,
             learning_rate, momentum, prior_y = None, normalize_online = False):
    '''
    Initialize a DAN with the specified parameters.

    Parameters
    ----------
    loss : str
        The type of loss function to use. Supported values are "supervised" and "unsupervised".
    beta_init : float
        The initial inverse temperature of the DAN.
    beta_reg : float
        The regularization on beta, represented with the character varsigma (ς) in the paper.
    softening : float
        Label softening or smoothing. If None, no softening is applied.
        Only used if loss is "supervised".
    number_features : int
        The number of features in the input data.
    number_memories : int
        The number of memories, or hidden units, in the DAN at the start of training.
    max_number_memories : int
        The maximum number of memories, or hidden units, that the DAN can hold. Only
        the first number_memories memories and their weights are used at the beginning
        of training. The rest are preallocated to be built later using splitting steepest descent.
    number_classes : int
        The number of classes.
    number_constraint_iterations : int
        The number of iterations of the Sinkhorn-Knopp algorithm to constrain the class weights
        of DAN_code.layers.LogDenseExp.
    learning_rate : float
        The learning rate of the optimizer.
    momentum : float
        The momentum of the optimizer.
    prior_y : tf.Tensor, optional
        A prior distribution over the classes. If None, a uniform prior is used.
        Defaults to None.
    normalize_online : bool, optional
        If True, normalize the inputs online. Otherwise, they are assumed to be already
        normalized, so the normalization is skipped. Defaults to False.
    Returns
    -------
    model : DAN
        A compiled DAN ready for training.
    '''
    # Set up the network structure
    model = DAN(number_preproc_layers = 0)
    
    model.add(Input(shape = (number_features,)))
    model.add(layers.Normalize(normalize_online))
    model.add(layers.DenseCor(number_memories, max_number_memories, beta_init, beta_reg))
    model.add(layers.LogDenseExp(number_constraint_iterations,
                                 max_number_memories, number_classes, prior_y))
    
    optimizer = SMD(learning_rate, momentum, momentum)
    
    if loss == "supervised":
        model.compile(optimizer, loss = losses.SupervisedNegLogLikelihood(softening),
                      metrics = ["accuracy"])
    elif loss == "unsupervised":
        model.compile(optimizer, loss = losses.UnsupervisedNegLogLikelihood(softening),
                      metrics = [])
    else:
        raise ValueError("Loss type not recognized. Supported values are 'supervised' and 'unsupervised'.")
    
    return model

def init_low_rank_DAN(loss, beta_init, beta_reg, softening, number_features, number_latent_features,
                      number_memories, max_number_memories, number_classes, number_constraint_iterations,
                      learning_rate, momentum, prior_y = None, normalize_online = False):
    '''
    Initialize a DAN with the specified parameters.

    Parameters
    ----------
    loss : str
        The type of loss function to use. Supported values are "supervised" and "unsupervised".
    beta_init : float
        The initial inverse temperature of the DAN.
    beta_reg : float
        The regularization on beta, represented with the character varsigma (ς) in the paper.
    softening : float
        Label softening or smoothing. If None, no softening is applied.
        Only used if loss is "supervised".
    number_features : int
        The number of features in the input data.
    number_memories : int
        The number of memories, or hidden units, in the DAN at the start of training.
    max_number_memories : int
        The maximum number of memories, or hidden units, that the DAN can hold. Only
        the first number_memories memories and their weights are used at the beginning
        of training. The rest are preallocated to be built later using splitting steepest descent.
    number_classes : int
        The number of classes.
    number_constraint_iterations : int
        The number of iterations of the Sinkhorn-Knopp algorithm to constrain the class weights
        of DAN_code.layers.LogDenseExp.
    learning_rate : float
        The learning rate of the optimizer.
    momentum : float
        The momentum of the optimizer.
    prior_y : tf.Tensor, optional
        A prior distribution over the classes. If None, a uniform prior is used.
        Defaults to None.
    normalize_online : bool, optional
        If True, normalize the inputs online. Otherwise, they are assumed to be already
        normalized, so the normalization is skipped. Defaults to False.
    Returns
    -------
    model : DAN
        A compiled DAN ready for training.
    '''
    # Set up the network structure
    model = DAN(number_preproc_layers = 0)
    
    model.add(Input(shape = (number_features,)))
    model.add(layers.Normalize(normalize_online))
    model.add(layers.DenseLowRankCor(number_latent_features, number_memories,
                                     max_number_memories, beta_init, beta_reg))
    model.add(layers.LogDenseExp(number_constraint_iterations,
                                 max_number_memories, number_classes, prior_y))
    
    optimizer = SMD(learning_rate, momentum, momentum)
    
    if loss == "supervised":
        model.compile(optimizer, loss = losses.SupervisedNegLogLikelihood(softening),
                      metrics = ["accuracy"])
    elif loss == "unsupervised":
        model.compile(optimizer, loss = losses.UnsupervisedNegLogLikelihood(),
                      metrics = [])
    else:
        raise ValueError("Loss type not recognized. Supported values are 'supervised' and 'unsupervised'.")
    
    return model

def calc_split_mask(x_train, y_train, model, max_number_eigvals, max_eigval, batch_size, adjust = True):
    '''
    Identify in a mask which DAN memories to split based on the minimum eigenvalues
    of the splitting matrices (see Appendix H). Assume the corresponding eigenvectors
    were learned as DAN_code.layers.DenseCor.eigvecs during the splitting phase of training,
    then the gradient of the loss with respect to the eigvectors is the eigenvectors
    rescaled by the eigenvalues, so we compute the eigenvalues as the dot product
    of the eigvectors and the rescaled eigvectors.

    Parameters
    ----------
    x_train : tf.Tensor
        The training data.
    y_train : tf.Tensor
        The training labels.
    model : DAN_code.models.DAN
        The DAN model to analyze.
    max_number_eigvals : int
        The maximum number of eigenvalues to split.
    max_eigval : float
        The maximum eigenvalue to split.
    batch_size : int
        The batch size for processing the training data.
    adjust : bool, optional
        Whether to run one step of power iteration to make the eigenvalues and eigenvectors
        slightly more accurate. Defaults to True.
    Returns
    -------
    mask : tf.Tensor
        A boolean mask indicating which memories to split. Given the maximum number of
        memories max_number_memories and the current number of memories number_memories of the DAN,
        the min(max_number_eigvals, number_memories, max_number_memories - number_memories)
        most negative eigenvalues below max_eigval are split.
    '''
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
    
    max_number_eigvals = np.min([max_number_eigvals, number_memories, max_number_memories - number_memories])
    mask = eigvals < k.minimum(func.kth_min(eigvals, k = max_number_eigvals + 1), max_eigval)
    
    return mask

def acc_scaled_eigvecs(x_batch, y_batch, running_weights, eigvecs, scaled_eigvecs, model):
    '''
    Accumulate eigenvectors scaled by their corresponding eigenvalues for a batch of data.
    The scaled eigenvectors scaled_eigvecs are the gradient of the model loss with respect
    to the eigenvectors eigvecs (see Appendix H). Running this function in a for loop
    on the entire training set like in calc_split_mask gives the average scaled eigenvectors,
    which can be used to compute the eigenvalues.

    Parameters
    ----------
    x_batch : tf.Tensor
        A batch of input data.
    y_batch : tf.Tensor
        A batch of labels corresponding to the input data.
    running_weights : tf.Variable
        A variable to keep track of the total number of samples processed so far.
    eigvecs : tf.Variable
        The eigenvectors of the DAN layer.
    scaled_eigvecs : tf.Variable
        A variable to accumulate the scaled eigenvectors.
    model : DAN_code.models.DAN
        The DAN model being analyzed.
    Returns
    -------
    L_batch : tf.Tensor
        The loss computed for the batch of data.
    '''
    running_weights.assign_add(x_batch.shape[0])
    with tf.GradientTape() as tape:
        tape.watch(eigvecs)
        h_pred = model(x_batch)
        
        L_batch = model.compiled_loss(y_batch, h_pred)
        
        scaled_eigvecs.assign_add(x_batch.shape[0]/running_weights * (tape.gradient(L_batch, eigvecs) - scaled_eigvecs))

    return L_batch
    
def compile_memorization_phase(model):
    '''
    Compile the DAN model in the memorization phase of training, where the memories
    of DAN_code.layers.DenseCor and the class weights of DAN_code.layers.LogDenseExp
    are trained, while the eigvecs of DAN_code.layers.DenseCor are fixed.

    Parameters
    ----------
    model : DAN
        The DAN model to compile.
    '''
    model.get_DAN_layer(1).kernel._trainable = True
    try:
        model.get_DAN_layer(1).basis._trainable = True
    except AttributeError:
        pass
    model.get_DAN_layer(1).beta._trainable = True
    model.get_DAN_layer(1).eigvecs._trainable = False
    model.get_DAN_layer(2).kernel._trainable = True
    
    model.compile(optimizer = model.optimizer, loss = model.loss, metrics = ["accuracy"])

def compile_splitting_phase(model):
    '''
    Compile the DAN model in the splitting phase of training, where the eigvecs of
    DAN_code.layers.DenseCor are trained, while the memories of DAN_code.layers.DenseCor
    and the class weights of DAN_code.layers.LogDenseExp are fixed.

    Parameters
    ----------
    model : DAN
        The DAN model to compile.
    '''
    model.get_DAN_layer(1).kernel._trainable = False
    try:
        model.get_DAN_layer(1).basis._trainable = False
    except AttributeError:
        pass
    model.get_DAN_layer(1).beta._trainable = False
    model.get_DAN_layer(1).eigvecs._trainable = True
    model.get_DAN_layer(2).kernel._trainable = False
    
    model.compile(optimizer = model.optimizer, loss = model.loss,
                  metrics = [metrics.RayleighQuotient(model)])

def train_vanilla_net(x_train, y_train, model, number_epochs, batch_size, verbose = False):
    callback_list = []
    if verbose == True:
        callback_list.append(TerminateOnNaN())
        
    # Train model
    model.fit(x_train, y_train, epochs = number_epochs, batch_size = batch_size,
              verbose = verbose, callbacks = callback_list, validation_split = 0.1)
    
    return model

def train_DAN(x_train, y_train, model, number_epochs, batch_size, patience = 0,
              training_phase = "memorization", record = None, verbose = False,
              name = "DAN", validation_split = 0.1):
    '''
    Train a DAN with the specified parameters.

    Parameters
    ----------
    x_train : tf.Tensor
        The training data.
    y_train : tf.Tensor
        The training labels.
    model : DAN_code.models.DAN
        The DAN model to train.
    number_epochs : int
        The number of training epochs.
    batch_size : int
        The batch size for training.
    patience : int, optional
        The number of epochs to wait for improvement in the validation loss before
        stopping training early. If None, no early stopping is applied. Defaults to 0.
    training_phase : str, optional
        The training phase to compile the model for. Supported values are "memorization"
        and "splitting". In the memorization phase, the DAN learns the memories of the
        DAN_code.layers.DenseCor and the class weights of DAN_code.layers.LogDenseExp.
        In the splitting phase, the eigvecs of DAN_code.layers.DenseCor are trained.
        Defaults to "memorization".
    record : str, optional
        The type of data to record during training. Supported values are "movies",
        "weights_with_splitting", "weights_without_splitting", "transitions" and None.
        If None, no data is recorded. If "transitions", the approximate average transition
        matrix between the classes of the memories is saved. Otherwise, memories are saved.
        If "movies", the all of the corresponding class weights of the DAN are also saved
        to make a movie of the training process. Otherwise, only the classes of the memories
        are saved. Defaults to None.
    verbose : bool, optional
        If True, training progress is printed to the console. Defaults to False.
    name : str, optional
        The name of the DAN model, used to save the trained model and the recorded data.
        Defaults to "DAN".
    validation_split : float, optional
        The fraction of the training data to use for validation. If 0, no validation is
        performed. Defaults to 0.1.
    Returns
    -------
    model : DAN
        The trained DAN model.
    '''  
    beta_reg = model.get_DAN_layer(1).beta_reg
    
    number_memories = model.get_DAN_layer(1).output_size
    
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
    
    if model.get_DAN_layer(1).beta_reg is None:
        model.get_DAN_layer(1).beta._trainable = False
    
    callback_list = []
    if patience is not None:
        #callback_list.append(EarlyStopping(monitor = monitored_quantity, patience = patience,
        #                                   mode = "min", start_from_epoch = number_epochs//2))
        callback_list.append(EarlyStopping(monitor = monitored_quantity, patience = patience, mode = "min"))
    
    if verbose == True:
        callback_list.append(TerminateOnNaN())
    
    if record == None:
        pass
    elif record == "movies":
        P = 25
        callback_list.append(callbacks.WeightEvolution(beta_reg, P, name, "for_movies"))
    elif (record == "weights_with_splitting") | (record == "weights_without_splitting"):
        # 2000 is a safety upper bound on the number of memories to save.
        P = np.minimum(2000, number_memories)
        callback_list.append(callbacks.WeightEvolution(beta_reg, P, name, record[8 :]))
    elif record == "beta":
        callback_list.append(callbacks.BetaEvolution(name))
    elif record == "transitions":
        transition_matrix = np.zeros((model.output_shape[-1], model.output_shape[-1]))
        callback_list.append(callbacks.AverageTransitionMatrix(transition_matrix))
    else:
        raise ValueError("Record type not supported. Expected 'movies', 'weights_with_splitting', 'weights_without_splitting' or 'transitions'.")
    
    # Train model
    model.fit(x_train, y_train, epochs = number_epochs, batch_size = batch_size,
              verbose = verbose, callbacks = callback_list, validation_split = validation_split)
    
    if record == "movies":
        w_list = load_contents("./Data/Weights/%s_w_with_beta_reg=%s_and_for_movies.npy" % (name, str(beta_reg)))
        g_list = load_contents("./Data/Weights/%s_g_with_beta_reg=%s_and_for_movies.npy" % (name, str(beta_reg)))
        
        animate(w_list, g_list, model)
    
    elif record == "transitions":
        with open("./Data/%s_transition_matrix_with_beta_reg=%s_and_%s_memories.npy" % (name, str(beta_reg), str(number_memories)), "wb") as f:
            np.fill_diagonal(transition_matrix, np.nan)
            transition_matrix = transition_matrix.T
            np.save(f, transition_matrix)
    
    return model

def train_DAN_with_an_annealing_schedule(x_train, y_train, model, number_epochs, number_annealing_epochs,
                                         batch_size, learning_rate_decay = 0, beta_final = None, slope = 1, patience = 0,
                                         training_phase = "memorization", record = None, verbose = False,
                                         name = "DAN", validation_split = 0.1):
    '''
    Train a DAN with the specified parameters. Increase the inverse temperature according
    to the annealing schedule in the callbacks.py module. Using an annealing schedule takes
    a bit more fine tuning than the effective loss.

    Parameters
    ----------
    x_train : tf.Tensor
        The training data.
    y_train : tf.Tensor
        The training labels.
    model : DAN_code.models.DAN
        The DAN model to train.
    number_epochs : int
        The number of training epochs.
    number_annealing_epochs : int
        The number of epochs to anneal the inverse temperature beta from its initial value
        beta_init to the final value beta_final. If 0, no annealing is performed
        and beta_final is set to beta_init.
    batch_size : int
        The batch size for training.
    beta_final : float, optional
        The final inverse temperature of the DAN after annealing. If None, it is set to
        the initial value beta_init, which is set during initialization. Defaults to None.
    slope : float, optional
        The slope of the power-law annealing schedule. Defaults to 1.
    patience : int, optional
        The number of epochs to wait for improvement in the validation loss before
        stopping training early. If None, no early stopping is applied. Defaults to 0.
    training_phase : str, optional
        The training phase to compile the model for. Supported values are "memorization"
        and "splitting". In the memorization phase, the DAN learns the memories of the
        DAN_code.layers.DenseCor and the class weights of DAN_code.layers.LogDenseExp.
        In the splitting phase, the eigvecs of DAN_code.layers.DenseCor are trained.
        Defaults to "memorization".
    record : str, optional
        The type of data to record during training. Supported values are "movies",
        "weights_with_splitting", "weights_without_splitting", "transitions" and None.
        If None, no data is recorded. If "transitions", the approximate average transition
        matrix between the classes of the memories is saved. Otherwise, memories are saved.
        If "movies", the all of the corresponding class weights of the DAN are also saved
        to make a movie of the training process. Otherwise, only the classes of the memories
        are saved. Defaults to None.
    verbose : bool, optional
        If True, training progress is printed to the console. Defaults to False.
    name : str, optional
        The name of the DAN model, used to save the trained model and the recorded data.
        Defaults to "DAN".
    validation_split : float, optional
        The fraction of the training data to use for validation. If 0, no validation is
        performed. Defaults to 0.1.
    Returns
    -------
    model : DAN
        The trained DAN model.
    '''
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
    
    model.get_DAN_layer(1).beta._trainable = False
    callback_list = []
    
    if learning_rate_decay != 0:
        inverse_time_decay = InverseTimeDecay(initial_learning_rate = model.optimizer.learning_rate, decay_steps = 1, decay_rate = learning_rate_decay)
        callback_list.append(LearningRateScheduler(inverse_time_decay))
    
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
        w_list = load_contents("./Data/Weights/%s_w_with_beta_reg=%s_and_for_movies.npy" % (name, str(beta_final)))
        g_list = load_contents("./Data/Weights/%s_g_with_beta_reg=%s_and_for_movies.npy" % (name, str(beta_final)))
        
        animate(w_list, g_list, model)
    
    elif record == "transitions":
        with open("./Data/%s_transition_matrix_with_beta_reg=%s_and_%s_memories.npy" % (name, str(beta), str(number_memories)), "wb") as f:
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
def save_DAN(model, beta_reg, number_memories, number_splits, name = "DAN"):
    '''
    Save a DAN to a file. The parameters other than the model itself are used to name
    the file so that it can be loaded with the same parameters.

    Parameters
    ----------
    model : DAN_code.models.DAN
        The DAN model to save.
    beta_reg : float
        The regularization on beta, represented with the character varsigma (ς) in the paper.
    number_memories : int
        The number of memories of the DAN.
    number_splits : int
        The number of splits for which the DAN was trained.
    name : str, optional
        The name of the model to save. Defaults to "DAN".
    '''
    model.save("./Data/Nets/%s_with_beta_reg=%s_and_%s_memories_for_%s_splits.keras"
               % (name, str(beta_reg), str(number_memories), str(number_splits)))

### load a trained DAN.
def load_DAN(beta_reg, number_memories, number_splits, custom_objects = custom_objects, name = "DAN"):
    '''
    Load a DAN from a file. The parameters other than custom_objects are used to find
    the name of the file to load.

    Parameters
    ----------
    beta_reg : float
        The regularization on beta, represented with the character varsigma (ς) in the paper.
    number_memories : int
        The number of memories of the DAN.
    number_splits : int
        The number of splits for which the DAN was trained.
    custom_objects : dict
        A dictionary of custom objects to use when loading the model. It is necessary
        because the DAN uses custom objects (layers, constraints, etc.) that are not part
        of the standard keras library. Defaults to the custom objects imported in this module.
    name : str, optional
        The name of the model to load. Defaults to "DAN".
    Returns
    -------
    model : DAN
        The loaded DAN model.
    '''
    model = load_model("./Data/Nets/%s_with_beta_reg=%s_and_%s_memories_for_%s_splits.keras"
                       % (name, str(beta_reg), str(number_memories), str(number_splits)),
                       custom_objects = custom_objects)
    return model

def load_contents(filename):
    '''
    Load contents from a file that was saved using np.save in append mode. Read the file
    until a ValueError is raised, which indicates that there are no more arrays to read.
    Used to read weights saved with the callbacks.WeightEvolution callback.

    Parameters
    ----------
    filename : str
        The path to the file to load.
    Returns
    -------
    contents : list
        List of numpy arrays saved in the file.
    '''
    contents = []
    
    with open(filename, "rb") as f:
        while True:
            try:
                contents.append(np.load(f))
            except ValueError:
                break
    
    return contents