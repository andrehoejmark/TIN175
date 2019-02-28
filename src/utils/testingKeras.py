
# Hello. This just my testing file for learning Keras. You can try using it if
# you want to, however, you will have to merge the dataset yourself.

import matplotlib.pyplot as plt

import keras
import numpy
import os
import sys

import tensorflow

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

from sklearn.preprocessing import MinMaxScaler

from loadCSV import loadInterpolatedCSV, separateCSV, loadConfigCSV

## 
## You can fiddle with these values in order to change the simulation.
## 

START_LEARNING_RATE = 1e-3 # The initial learning rate.
LOSS_FACTOR = 0.1 # The change of learning rate when unsuccessful in improving the result.
MIN_LEARNING_RATE = 1e-4 # The absolute minimum learning rate.
LEARNING_PATIENCE = 0 # The patience to have when not able to improve the learning rate (0 is pretty good to avoid overfitting).
NUM_EPOCHS = 10
STEPS_PER_EPOCH = 64
EARLY_STOP_PATIENCE = 0
WARMUP_STEPS = 100 # Number to steps before exiting the warm up phase.
NUM_GATED_REOCCURRING_UNITS = 512
ACTIVATION_FUNCTION = 'sigmoid'
TIME_SHIFT_IN_HOURS = 12
TRAINING_SPLITTING = 0.9
SHOW_OUTPUT_AFTER_SIM = True

READ_BATCH_SIZE = 16
# READ_SEQUENCE_LENGTH = 24 * 30 * 8 # Read eight months of data.
READ_SEQUENCE_LENGTH = 24 * 7 # Read a puny fraction of the data.
NETWORK_TYPE = 'GRU'

PLOT_OUTPUT_FOLDER = "../../simulation/"
PLOT_OUTPUT_SUB_NAME = ""
READ_CSV_FILE = "merged_2018_from_oct.csv"

CONFIG_LOCATION = "../../simulation/hyperparameters.csv"
LOADED_CONFIG_FILE = None

def loadConfigFile(name):
  global LOADED_CONFIG_FILE
  LOADED_CONFIG_FILE = loadConfigCSV(CONFIG_LOCATION)

def applyConfig(conf):
  global START_LEARNING_RATE
  global LOSS_FACTOR
  global MIN_LEARNING_RATE
  global LEARNING_PATIENCE
  global NUM_EPOCHS
  global STEPS_PER_EPOCH
  global EARLY_STOP_PATIENCE
  global WARMUP_STEPS
  global NUM_GATED_REOCCURRING_UNITS
  global ACTIVATION_FUNCTION
  global TIME_SHIFT_IN_HOURS
  global TRAINING_SPLITTING
  global READ_BATCH_SIZE
  global READ_SEQUENCE_LENGTH
  global PLOT_OUTPUT_SUB_NAME
  global NETWORK_TYPE
  TIME_SHIFT_IN_HOURS = conf.time_shift
  TRAINING_SPLITTING = conf.training_split
  START_LEARNING_RATE = conf.start_learning_rate
  LOSS_FACTOR = conf.loss_factor
  MIN_LEARNING_RATE = conf.min_learning_rate
  LEARNING_PATIENCE = conf.learning_patience
  NUM_EPOCHS = conf.num_epochs
  STEPS_PER_EPOCH = conf.steps_per_epoch
  EARLY_STOP_PATIENCE = conf.early_stop_patience
  WARMUP_STEPS = conf.warmup_steps
  NUM_GATED_REOCCURRING_UNITS = conf.num_gru
  ACTIVATION_FUNCTION = conf.activation_function
  NETWORK_TYPE = conf.network_type
  PLOT_OUTPUT_SUB_NAME = "%s_" % str(conf.id)

def runConfigs():
  global SHOW_OUTPUT_AFTER_SIM
  if LOADED_CONFIG_FILE:
    cid = 0
    conf = None
    while True:
      conf = LOADED_CONFIG_FILE.getConfig(cid)
      if conf:
        SHOW_OUTPUT_AFTER_SIM = False
        applyConfig(conf)
        performSimulation("simulate")
        cid = cid + 1
      else:
        print("Ended config run at cid: %d" % cid)
        break
  else:
    print("No config CSV file has been loaded.")

def savePlotToFile(name):
  plt.savefig("%s/%s.png" % (PLOT_OUTPUT_FOLDER, name), bbox_inches = "tight")

#
# Load data from CSV files.
#

data = loadInterpolatedCSV(READ_CSV_FILE)

if not data:
  sys.exit(1)

target_names = ["Gothenburg temperature", "Gothenburg wind direction", "Gothenburg wind speed", "Gothenburg pressure"]
num_target_columns = len(target_names)

def performSimulation(cmd):
  (smhi_in, smhi_out) = separateCSV(data, 6, num_target_columns)
  time_shift_hours = TIME_SHIFT_IN_HOURS
  training_split = TRAINING_SPLITTING
  # Perform time shift.
  smhi_out.shiftDataColumns(-time_shift_hours)
  smhi_in.drop(time_shift_hours)
  smhi_out.pop(time_shift_hours)
  # Transform date format
  in_values = smhi_in.valuesWithDate()
  out_values = smhi_out.valuesWithoutDate()
  # print(in_values)
  print(in_values.shape)
  # print(out_values)
  print(out_values.shape)
  # 
  # Visulisation.
  # 
  def plotColumn(header, y_axis_unit, rows, column_idx):
    column = []
    for row in rows:
      column.append(row[column_idx])
    plt.figure(figsize=(15,5))
    plt.plot(column, label=header)
    plt.ylabel(y_axis_unit)
    plt.legend()
    if cmd == "view_input_plots":
      plt.show()
    else:
      savePlotToFile("input_plot_%s" % header)
  # Parse command.
  if cmd == "view_input_plots" or cmd == "save_input_plots":
    units = ["Temperature [C]", "Wind direction [deg]", "Wind speed [m/s]", "Pressure [kilo Pascal]"]
    for i in range(2, len(smhi_in.header)):
      plotColumn(smhi_in.header[i], units[(i - 2) % len(units)], in_values, i + 4)
  elif cmd == "simulate":
    # Split and normalize data.
    num_train = int(len(in_values) * training_split)
    in_train = in_values[0:num_train]
    in_test = in_values[num_train:]
    out_train = out_values[0:num_train]
    out_test = out_values[num_train:]
    in_scaler = MinMaxScaler()
    in_train_scaled = in_scaler.fit_transform(in_train)
    in_test_scaled = in_scaler.transform(in_test)
    out_scaler = MinMaxScaler()
    out_train_scaled = out_scaler.fit_transform(out_train)
    out_test_scaled = out_scaler.transform(out_test)
    ## 
    ## The following is a modified adaptation of the tutorial code.
    ## 
    print(READ_SEQUENCE_LENGTH)
    num_in_signals = in_values.shape[1]
    num_out_signals = out_values.shape[1]
    def batch_generator(batch_size, sequence_length):
      "Generator function for creating random batches of training-data."
      while True:
        in_shape = (batch_size, sequence_length, num_in_signals)
        in_batch = numpy.zeros(shape=in_shape, dtype=numpy.float16)
        out_shape = (batch_size, sequence_length, num_out_signals)
        out_batch = numpy.zeros(shape=out_shape, dtype=numpy.float16)
        for i in range(batch_size):
          idx = numpy.random.randint(num_train - sequence_length)
          in_batch[i] = in_train_scaled[idx:idx+sequence_length]
          out_batch[i] = out_train_scaled[idx:idx+sequence_length]
          yield (in_batch, out_batch)
    generator = batch_generator(READ_BATCH_SIZE, READ_SEQUENCE_LENGTH)
    in_batch, out_batch = next(generator)
    print(in_batch.shape)
    print(out_batch.shape)
    validation_data = (
      numpy.expand_dims(in_test_scaled, axis=0),
      numpy.expand_dims(out_test_scaled, axis=0))
    # Setup network.
    model = Sequential()
    
    # TODO: Set the network type depending on 'NETWORK_TYPE'
    if NETWORK_TYPE == 'GRU':
      model.add(
        GRU(units=NUM_GATED_REOCCURRING_UNITS,
          return_sequences=True, input_shape=(None, num_in_signals,)))
    else:
      print("Don't know what the network type '%s' is." % NETWORK_TYPE)
      return False
    model.add(Dense(num_out_signals, activation=ACTIVATION_FUNCTION))
    warmup_steps = WARMUP_STEPS
    
    def loss_mse_warmup(out_true, out_pred):
      "Calculate the Mean Squared Error"
      out_true_slice = out_true[:, warmup_steps:, :]
      out_pred_slice = out_pred[:, warmup_steps:, :]
      loss = tensorflow.losses.mean_squared_error(labels=out_true_slice, predictions=out_pred_slice)
      loss_mean = tensorflow.reduce_mean(loss)
      return loss_mean
    optimizer = RMSprop(lr=START_LEARNING_RATE)
    model.compile(loss=loss_mse_warmup, optimizer=optimizer)
    model.summary()
    path_checkpoint = 'weather_sim_checkpoint.keras'
    callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
      monitor='val_loss',
      verbose=1,
      save_weights_only=True,
      save_best_only=True)
    callback_early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE, verbose=1)
    callback_tensorboard = TensorBoard(log_dir='./keras_logs/', histogram_freq=0, write_graph=False)
    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
      factor=LOSS_FACTOR,
      min_lr=MIN_LEARNING_RATE,
      patience=LEARNING_PATIENCE,
      verbose=1)
    callbacks = [callback_early_stopping, callback_checkpoint, callback_tensorboard, callback_reduce_lr]
    print("Performing simulation...")
    model.fit_generator(generator=generator,
      epochs=NUM_EPOCHS,
      steps_per_epoch=STEPS_PER_EPOCH,
      validation_data=validation_data,
      callbacks=callbacks)
    try:
        model.load_weights(path_checkpoint)
    except Exception as error:
        print("Error trying to load checkpoint: %s" % error)
        return False
    result = model.evaluate(
      x=numpy.expand_dims(in_test_scaled, axis=0),
      y=numpy.expand_dims(out_test_scaled, axis=0))
    print("loss (test-set):", result)
    # Do some comparison plotting.
    def plot_comparison(start_idx, length=100, train=True):
      "Plot the predicted and true output-signals."
      if train:
        inp = in_train_scaled
        out_true = out_train
      else:
        inp = in_test_scaled
        out_true = out_test
      end_idx = start_idx + length
      inp = inp[start_idx:end_idx]
      out_true = out_true[start_idx:end_idx]
      inp = numpy.expand_dims(inp, axis=0)
      out_pred = model.predict(inp)
      out_pred_rescaled = out_scaler.inverse_transform(out_pred[0])
      for signal in range(0, num_target_columns):
        signal_pred = out_pred_rescaled[:, signal]
        signal_true = out_true[:, signal]
        plt.figure(figsize=(15,5))
        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')
        p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)
        header = target_names[signal]
        plt.ylabel(header)
        plt.legend()
        if SHOW_OUTPUT_AFTER_SIM:
          plt.show()
        else:
          savePlotToFile("output_plot_%s%s" % (PLOT_OUTPUT_SUB_NAME, header))
    # Render comparison graphs.
    plot_comparison(start_idx=0, length=480, train=True)
  else:
    print("Donno what \"%s\" is. You can use 'view_input_plots', 'save_input_plots', 'run_config' or 'simulate' at the moment." % cmd)
    return False
  return True

while True:
  print("Input your command (or help): ")
  cmd = input()
  if cmd == "exit":
    break
  elif cmd == "help":
    print("Help yourself.")
  elif cmd == "run_config":
    loadConfigFile(CONFIG_LOCATION)
    runConfigs()
  else:
    performSimulation(cmd)
