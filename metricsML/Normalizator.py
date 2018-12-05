from metricsML.NormalizationType import NormalizationType
import numpy as np
import math

def normalization(normalization, train_data, test_data, validation_data=None):
    if not isinstance(normalization, NormalizationType):
        print("Unknown normalization specified, use " + str(NormalizationType.PERCENTAGE) + " for normalizing data")
        normalization = NormalizationType.PERCENTAGE;
    
    if (normalization is NormalizationType.PERCENTAGE):
        __percentageNormalization(train_data, test_data, validation_data)
    elif (normalization is NormalizationType.LOGARITHM):
        __logarithmNormalization(train_data, test_data, validation_data)
    else:
        raise TypeError("Unhandled normalization selected")


def __percentageNormalization(train_data, test_data, validation_data=None):
    nColumns = train_data.shape[1] if len(train_data.shape) == 2 else 0;
    train_max = np.amax(train_data, axis=0)
    test_max = np.amax(test_data, axis=0)
    if (validation_data is not None):
        validation_max =  np.amax(validation_data, axis=0)
    else:
        validation_max = np.zeros(nColumns)
    
    max_vector = np.amax([train_max, test_max, validation_max], axis=0)
    
    train_data = train_data/max_vector
    test_data = test_data/max_vector
    
    if (validation_data is not None):
        validation_data = validation_data/max_vector
        
def __logarithmNormalization(train_data, test_data, validation_data=None):
    nColumns = train_data.shape[1] if len(train_data.shape) == 2 else 0;
    train_max = np.amax(train_data, axis=0)
    test_max = np.amax(test_data, axis=0)
    if (validation_data is not None):
        validation_max =  np.amax(validation_data, axis=0)
    else:
        validation_max = np.zeros(nColumns)
    
    max_vector = np.amax([train_max, test_max, validation_max], axis=0)
    
    for column in range(0, nColumns):
        max_value = max_vector[column]
        if (max_value > 1):
            train_data[:, column] = [__positiveLogarithm(x) for x in train_data[:, column]]
            test_data[:, column] = [__positiveLogarithm(x) for x in test_data[:, column]]
            if (validation_data is not None):
                validation_data[:, column] = [__positiveLogarithm(x) for x in validation_data[:, column]]

def __positiveLogarithm(number, base):
    if (number > 1):
        return math.log(number, base)
    else:
        return 0