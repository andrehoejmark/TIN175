
#
# Author: Felix Hulthén.
#

"""
A container script for holding the 'SimulationData' class.
"""

import sys

from src.utils.loadCSV import separateCSV, deleteCSVColumn
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, LSTM, CuDNNLSTM
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import tensorflow
import numpy
import keras

from src.utils.model import loadModel

class SimulationData:
    """
    A class for holding a Keras RNN, and accessing its data fields.
    """
    def __init__(self, hyperparams, headers, data=None):
        hyperparams.getConfigString()
        START_COLUMN_IDX = 6
        (smhi_in, smhi_out) = separateCSV(data, START_COLUMN_IDX, hyperparams.outputs)
        if hyperparams.keep_city_as_input:
          smhi_in = data
        smhi_in.use_full_timestamp = hyperparams.use_full_timestamp
        i = len(smhi_out.header) - 1
        while i >= 0:
          print("Comaprint \"%s\" vs \"%s\" (%s)" % (smhi_out.header[i], headers, str(smhi_out.header[i] in headers)))
          if not (smhi_out.header[i] in headers):
            deleteCSVColumn(smhi_out, i)
          i = i - 1
        print("Size of %d" % len(smhi_out.header))
        print("Size of row %d" % len(smhi_out.data[0]))
        self.id = "Data_" + hyperparams.plot_output_sub_name
        self.smhi_in = smhi_in
        self.smhi_out = smhi_out
        self.time_shift_hours = hyperparams.time_shift_in_hours
        self.training_split = hyperparams.training_splitting
        self.loss_function = "mse"

        # Perform time shift.
        self.smhi_out.shiftDataColumns(- self.time_shift_hours)
        self.smhi_in.drop(self.time_shift_hours)
        self.smhi_out.pop(self.time_shift_hours)


        # Transform date format
        self.in_values=self.smhi_in.valuesWithDate()
        #  self.out_values=smhi_out.valuesWithoutDate()
        self.out_values=self.smhi_out.values()
        self.num_in_signals = self.in_values.shape[1]
        self.num_out_signals = self.out_values.shape[1]

        print("Num inputs: %d Num outputs: %d" % (self.num_in_signals, self.num_out_signals))

        self.model = None

    def splitAndNormalize(self):
        """
        splitAndNormalize : Will split the held datasets and normalize the input and
                            output in order to comply with the used activation function.
        """
        self.num_train=int(len(self.in_values)*self.training_split)
        val_len = 720
        self.in_train=self.in_values[0:self.num_train]
        self.in_test=self.in_values[self.num_train:-val_len]
        self.in_val=self.in_values[-val_len:]

        self.out_train=self.out_values[0:self.num_train]
        self.out_test=self.out_values[self.num_train:-val_len]
        self.out_val=self.out_values[-val_len:]

        self.in_scaler=MinMaxScaler()
        self.in_train_scaled=self.in_scaler.fit_transform(self.in_train)
        self.in_test_scaled=self.in_scaler.transform(self.in_test)
        self.in_val_scaled=self.in_scaler.transform(self.in_val)
        self.out_scaler=MinMaxScaler()
        self.out_train_scaled=self.out_scaler.fit_transform(self.out_train)
        self.out_test_scaled=self.out_scaler.transform(self.out_test)
        self.out_val_scaled=self.out_scaler.transform(self.out_val)



    def createValidationData(self):
        """
        createValidationData : Create a validation data object.
        """
        self.validation_data=(
            numpy.expand_dims(self.in_test_scaled, axis=0),
            numpy.expand_dims(self.out_test_scaled, axis=0))


    def getBatchGenerator(self, hyperparams):
        """
        getBatchGenerator : Creates a batch generator for use during later training.
          hyperprarams : The hyperparams object that dictates how the RNN is costructed.
        """
        batch_size=hyperparams.read_batch_size
        sequence_length=336  #  hyperparams.read_sequence_length
        while True:
            in_shape=(batch_size, sequence_length, self.num_in_signals)
            in_batch=numpy.zeros(shape=in_shape, dtype=numpy.float16)
            out_shape=(batch_size, sequence_length, self.num_out_signals)
            out_batch=numpy.zeros(shape=out_shape, dtype=numpy.float16)
            for i in range(batch_size):
                if self.num_train < sequence_length:
                    print("Error: Num train %d and num sequence %d" % (self.num_train, sequence_length))
                idx = numpy.random.randint(self.num_train - sequence_length)
                in_batch[i] = self.in_train_scaled[idx: idx + sequence_length]
                out_batch[i] = self.out_train_scaled[idx: idx + sequence_length]
                yield (in_batch, out_batch)


    def lossMSEWarmup(self, out_true, out_pred):
        """
        lossMSEWarmup : Calculates the mean square error over the trained network.
        """
        #  "Calculate the Mean Squared Error"
        out_true_slice=out_true[:, self.warmup_steps:, :]
        out_pred_slice=out_pred[:, self.warmup_steps:, :]
        self.loss=tensorflow.losses.mean_squared_error(labels=out_true_slice, predictions=out_pred_slice)
        self.loss_mean=tensorflow.reduce_mean(self.loss)
        return self.loss_mean

    def setupAndPerformSimulation(self, hyperparams):
        """
        setupAndPerformSimulation : Will setup the network according to the provided hyperparams object.
          hyperprarams : The hyperparams object that dictates how the RNN is costructed.
        """
        if hyperparams.network_type == "tutorial_model":
          self.model = loadModel(hyperparams=hyperparams, model_name="tutorial_model", in_shape=self.num_in_signals, out_shape=self.num_out_signals)
        else:
          self.model = Sequential()
          if hyperparams.network_type == "GRU":
              self.model.add(
                  GRU(units=hyperparams.num_gated_reoccurring_units,
                      return_sequences=True, input_shape=(None, self.num_in_signals,)))
          elif hyperparams.network_type == "LSTM":
              # self.model.add(LSTM(hyperparams.num_gated_reoccurring_units, return_sequences = True))
              self.model.add(
                  LSTM(units=hyperparams.num_gated_reoccurring_units,
                    return_sequences=True, input_shape=(None, self.num_in_signals,)))
          elif hyperparams.network_type == "CuDNNLSTM":
              self.model.add(
                  CuDNNLSTM(units=hyperparams.num_gated_reoccurring_units,
                    return_sequences=True, input_shape=(None, self.num_in_signals,)))
          else:
              print("Don't know what the network type '%s' is." % hyperparams.network_type)
              sys.exit(1)

        self.model.add(Dense(self.num_out_signals, activation=hyperparams.activation_function))
        self.warmup_steps = hyperparams.warmup_steps

        optimizer = RMSprop(lr=hyperparams.start_learning_rate)
        self.model.compile(loss='mse', optimizer=optimizer)

        self.path_checkpoint = hyperparams.output_folder + 'weather_sim_checkpoint.keras'
        callback_checkpoint = ModelCheckpoint(filepath=self.path_checkpoint,
                                              monitor='val_loss',
                                              verbose=1,
                                              save_weights_only=True,
                                              save_best_only=True)
        callback_early_stopping = EarlyStopping(monitor='val_loss', patience=hyperparams.early_stop_patience, verbose=1)
        callback_tensorboard = TensorBoard(log_dir = hyperparams.output_folder + 'keras_logs/', histogram_freq=0, write_graph=False)
        callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                               factor=hyperparams.loss_factor,
                                               min_lr=hyperparams.min_learning_rate,
                                               patience=hyperparams.learning_patience,
                                               verbose=1)

        self.callbacks = [callback_early_stopping, callback_checkpoint, callback_tensorboard, callback_reduce_lr]

    def performSimulation(self, hyperparams):
        """
        performSimulation : requires that the other setup methods have
                            been called, and then performs a simulation
                            according to the hyperparameters object.
        """
        generator = self.getBatchGenerator(hyperparams)
        in_batch, out_batch = next(generator)
        self.model.summary()
        print("Performing simulation...")
        self.model.fit_generator(generator=generator,
                                 epochs=hyperparams.num_epochs,
                                 steps_per_epoch=hyperparams.steps_per_epoch,
                                 validation_data=self.validation_data,
                                 callbacks=self.callbacks)
