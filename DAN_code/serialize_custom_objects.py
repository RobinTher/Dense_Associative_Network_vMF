from DAN_code import functions
from DAN_code import callbacks
from DAN_code import constraints
from DAN_code import initializers
from DAN_code import layers
from DAN_code import losses
from DAN_code import metrics
from DAN_code.optimizers import SMD

from tensorflow.keras.utils import get_custom_objects

### Collect classes defined throughout all files of this module in a single dictionary
def collect_custom_objects(local_custom_objects = {}):
    '''
    Collects the custom objects defined in DAN_code and returns them as a dictionary.

    Parameters
    ----------
    local_custom_objects : dict, optional
        A dictionary containing additional custom objects defined in the local scope.
        Used in DAN_code.models to avoid circular imports. Can also be used to add
        custom objects from other modules.

    Returns
    -------
    custom_objects : dict
        A dictionary containing the custom objects defined in the DAN_code module.
        Should be updated when adding new custom objects.
    '''
    custom_objects = {"log_gamma_ratio" : functions.log_gamma_ratio,
                      "unaveraged_rayleigh_quotient" : functions.unaveraged_rayleigh_quotient,
                      "BetaScheduler" : callbacks.BetaScheduler, "WeightEvolution" : callbacks.WeightEvolution,
                      "AverageTransitionMatrix" : callbacks.AverageTransitionMatrix, "UnitTwoNorm" : constraints.UnitTwoNorm,
                      "AltOneNorm" : constraints.AltOneNorm, "RandomSpherical" : initializers.RandomSpherical,
                      "Categorical" : initializers.Categorical, "Normalize" : layers.Normalize, "DenseCor" : layers.DenseCor,
                      "LogDenseExp" : layers.LogDenseExp, "RayleighQuotient" : metrics.RayleighQuotient,
                      "SupervisedNegLogLikelihood" : losses.SupervisedNegLogLikelihood,
                      "UnsupervisedNegLogLikelihood" : losses.UnsupervisedNegLogLikelihood, "SMD" : SMD}
    custom_objects.update(local_custom_objects)
    
    return custom_objects