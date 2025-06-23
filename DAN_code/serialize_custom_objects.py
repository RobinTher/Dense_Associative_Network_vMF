from DAN_code import functions
from DAN_code import callbacks
from DAN_code import constraints
from DAN_code import initializers
from DAN_code import layers
from DAN_code import losses
from DAN_code import metrics
from DAN_code.optimizers import SMD
from DAN_code import models

from tensorflow.keras.utils import get_custom_objects

### Collect classes defined throughout all files of this module in a single dictionary
def collect_custom_objects():
    
    custom_objects = {"log_gamma_ratio" : functions.log_gamma_ratio,
                      "unaveraged_rayleigh_quotient" : functions.unaveraged_rayleigh_quotient,
                      "BetaScheduler" : callbacks.BetaScheduler, "WeightEvolution" : callbacks.WeightEvolution,
                      "AverageTransitionMatrix" : callbacks.AverageTransitionMatrix,
                      "ToggleMetric" : callbacks.ToggleMetric, "UnitTwoNorm" : constraints.UnitTwoNorm,
                      "AltOneNorm" : constraints.AltOneNorm, "RandomSpherical" : initializers.RandomSpherical,
                      "VMFMixture" : initializers.VMFMixture, "Categorical" : initializers.Categorical,
                      "Normalize" : layers.Normalize, "DenseCor" : layers.DenseCor,
                      "LogDenseExp" : layers.LogDenseExp, "RayleighQuotient" : metrics.RayleighQuotient,
                      "NegLogLikelihood" : losses.NegLogLikelihood, "SupervisedNegLogLikelihood" : losses.SupervisedNegLogLikelihood,
                      "UnsupervisedNegLogLikelihood" : losses.UnsupervisedNegLogLikelihood, "SMD" : SMD, "DAN" : models.DAN}
    
    return custom_objects