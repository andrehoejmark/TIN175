
#
# Author: Rasmus Claesen.
#

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, LSTM, TimeDistributed, CuDNNLSTM
from tensorflow.python.keras.optimizers import RMSprop


############################################################################
#  Models defined as functions which creates and returns the model of choice
#  each model needs their own function but hyperparameters remain shared,
#  mostly just matches network type (GRU/LSTM) from hyperparams.
############################################################################


def tutorial_model(hyperparams=None, in_shape=None, out_shape=None):
    """
    :param hyperparams: Set of hyperparameters
    :param in_shape: Shape of model inputs
    :param out_shape: Shape of model outputs
    :return: A keras model based on the tutorial
    """
    model = Sequential()

    if hyperparams.network_type == 'LSTM':
        model = add_LSTM(model=model, units=hyperparams.num_gated_reoccurring_units, input_shape=(None, in_shape))

    elif hyperparams.network_type == 'GRU':
        model = add_GRU(model=model, units=hyperparams.num_gated_reoccurring_units, input_shape=(None, in_shape))

    elif hyperparams.network_type == 'CuDNNLSTM':
        model = add_GRU(model=model, units=hyperparams.num_gated_reoccurring_units, input_shape=(None, in_shape))


    model.add(Dense(out_shape, activation=hyperparams.activation_function))
    optimizer = RMSprop(lr=hyperparams.start_learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    return model


def IJCA_model(hyperparams):
    """
    :param hyperparams: Set of hyperparameters
    :return: A model based on Sequence to Sequence Weather Forecasting with Long
    Short-Term Memory Recurrent Neural Networks

    https://www.ijcaonline.org/archives/volume143/number11/zaytar-2016-ijca-910497.pdf
    """
    model = Sequential()

    if hyperparams.network_type == 'LSTM':
        model = add_LSTM(model=model, units=hyperparams.num_gated_reoccurring_units)
        model.add(Dense(100, activation=hyperparams.activation_function))
        model.add(LSTM(units=128, input_shape=(None, 24, 100), return_sequences=True))

    else:
        model = add_GRU(model=model, units=hyperparams.num_gated_reoccurring_units)
        model.add(Dense(100, activation=hyperparams.activation_function))
        model.add(GRU(units=128, input_shape=(None, 24, 100), return_sequences=True))

    model.add(Dense(4, activation=hyperparams.activation_function))
    optimizer=RMSprop(lr=hyperparams.start_learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    return model

def Sentdex_model(hyperparams):
    """
    :param hyperparams: Set of hyperparameters
    :return: A model based on RNN analysis of MNIST data by Sentdex

    https://pythonprogramming.net/
    """
    model = Sequential()

    if hyperparams.network_type == 'LSTM':
        model = add_LSTM(model=model, units=hyperparams.num_gated_reoccurring_units)
        model.add(LSTM(units=128, input_shape=(None, 24, 100), return_sequences=True))

    else:
        model = add_GRU(model=model, units=hyperparams.num_gated_reoccurring_units)
        model.add(GRU(units=128, input_shape=(None, 24, 100), return_sequences=True))

    model.add(Dense(64, activation=hyperparams.activation_function))
    model.add(Dense(4, activation=hyperparams.activation_function))
    optimizer=RMSprop(lr=hyperparams.start_learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    return model


#  Not the best of ways but it'll do for switching models from hyperparams
def add_LSTM(model=None, units=None, input_shape=None):
    model.add(
        LSTM(units=units,
            return_sequences=True, input_shape=input_shape))
    return model if not None else None


def add_GRU(model=None, units=None, input_shape=None):
    model.add(
        GRU(units=units,
            return_sequences=True, input_shape=input_shape))
    return model if not None else None

def add_CuDNNLSTM(model=None, units=None, input_shape=None):
    model.add(
        CuDNNLSTM(units=units,
            return_sequences=True, input_shape=input_shape))
    return model if not None else None


def loadModel(hyperparams=None, model_name="", in_shape=None, out_shape=None):
    """
    :param hyperparams: Set of hyperparameters
    :param model_name: Needs to match one of the defined models
    :param in_shape: input shape to pass on
    :param out_shape: output shape to pass on
    :return: A complete model of specified choice
    """
    model = None
    if model_name == "tutorial_model":
        model = tutorial_model(hyperparams, in_shape=in_shape, out_shape=out_shape)
    elif model_name == "IJCA_model":
        model = IJCA_model(hyperparams)
    elif model_name == "Sentdex_model":
        model = Sentdex_model(hyperparams)
    else:
        return None
    return model if not None else None


                                                                                                                        
