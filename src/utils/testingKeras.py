
# Hello. This just my testing file for learning Keras. You can try using if you want to, however, you will have
# to merge the dataset yourself.

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

from loadCSV import loadInterpolatedCSV, separateCSV

## 
## You can fiddle with these values in order to change the simulation.
## 

START_LEARNING_RATE = 1e-3 # The initial learning rate.
LOSS_FACTOR = 0.1 # The change of learning rate when unsuccessful in improving the result.
MIN_LEARNING_RATE = 1e-4 # The absolute minimum learning rate.
LEARNING_PATIENCE = 0 # The patience to have when not able to improve the learning rate (0 is pretty good to avoid overfitting).
NUM_EPOCHS = 5
STEPS_PER_EPOCH = 10
EARLY_STOP_PATIENCE = 5
WARMUP_STEPS = 10 # Number to steps before exiting the warm up phase.
NUM_GATED_REOCCURING_UNITS = 512
ACTIVATION_FUNCTION = 'sigmoid'

READ_CSV_FILE = "merged_2018.csv"

TIME_SHIFT_IN_HOURS = 12
TRAINING_SPLITTING = 0.9




data = loadInterpolatedCSV(READ_CSV_FILE)

if not data:
  sys.exit(1)

target_names = ["Gothenburg temperature", "Gothenburg wind direction", "Gothenburg wind speed", "Gothenburg pressure"]
num_target_columns = len(target_names)

(smhi_in, smhi_out) = separateCSV(data, 6, num_target_columns)
time_shift_hours = TIME_SHIFT_IN_HOURS
training_split = TRAINING_SPLITTING
smhi_out.shiftDataColumns(-time_shift_hours)

smhi_in.drop(time_shift_hours)
smhi_out.pop(time_shift_hours)

in_values = smhi_in.valuesWithDate()
out_values = smhi_out.valuesWithoutDate()

# print(in_values)
print(in_values.shape)

# print(out_values)
print(out_values.shape)

def plotColumn(header, y_axis_unit, rows, column_idx):
  column = []
  for row in rows:
    column.append(row[column_idx])
  plt.figure(figsize=(15,5))
  plt.plot(column, label=header)
  plt.ylabel(y_axis_unit)
  plt.legend()
  plt.show()

print("View plots [y/N]")

if input() is "y":
  units = ["Temperature [C]", "Wind direction [deg]", "Wind speed [m/s]", "Pressure [kilo Pascal]"]
  for i in range(2, len(smhi_in.header)):
    plotColumn(smhi_in.header[i], units[(i - 2) % len(units)], in_values, i + 4)

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
## Now I should be albe to use the tutorial training code :)
## 

proposed_batch_size = 16

proposed_sequence_length = 24 * 7 * 1 # Read 1 week of data.
print(proposed_sequence_length)

num_in_signals = in_values.shape[1]
num_out_signals = out_values.shape[1]

def batch_generator(batch_size, sequence_length):
  "Generator function for creating random batches of training-data."
  while True:
    # Allocate a new array for the batch of input-signals.
    in_shape = (batch_size, sequence_length, num_in_signals)
    in_batch = numpy.zeros(shape=in_shape, dtype=numpy.float16)
    # Allocate a new array for the batch of output-signals.
    out_shape = (batch_size, sequence_length, num_out_signals)
    out_batch = numpy.zeros(shape=out_shape, dtype=numpy.float16)
    # Fill the batch with random sequences of data.
    for i in range(batch_size):
      # Get a random start-index.
      # This points somewhere into the training-data.
      idx = numpy.random.randint(num_train - sequence_length)
      # Copy the sequences of data starting at this index.
      in_batch[i] = in_train_scaled[idx:idx+sequence_length]
      out_batch[i] = out_train_scaled[idx:idx+sequence_length]
      yield (in_batch, out_batch)

generator = batch_generator(proposed_batch_size, proposed_sequence_length)

in_batch, out_batch = next(generator)

print(in_batch.shape)
print(out_batch.shape)

validation_data = (
  numpy.expand_dims(in_test_scaled, axis=0),
  numpy.expand_dims(out_test_scaled, axis=0))

model = Sequential()
model.add(
  GRU(units=NUM_GATED_REOCCURING_UNITS,
    return_sequences=True, input_shape=(None, num_in_signals,)))

model.add(Dense(num_out_signals, activation=ACTIVATION_FUNCTION))

warmup_steps = WARMUP_STEPS

def loss_mse_warmup(out_true, out_pred):
  """
  Calculate the Mean Squared Error between y_true and y_pred,
  but ignore the beginning "warmup" part of the sequences.
  y_true is the desired output.
  y_pred is the model's output.
  """
  # The shape of both input tensors are:
  # [batch_size, sequence_length, num_y_signals].
  # Ignore warmup section
  out_true_slice = out_true[:, warmup_steps:, :]
  out_pred_slice = out_pred[:, warmup_steps:, :]
  # These sliced tensors both have this shape:
  # [batch_size, sequence_length - warmup_steps, num_y_signals]
  # Calculate the MSE loss for each value in these tensors.
  # This outputs a 3-rank tensor of the same shape.
  loss = tensorflow.losses.mean_squared_error(labels=out_true_slice, predictions=out_pred_slice)
  # Keras may reduce this across the first axis (the batch)
  # but the semantics are unclear, so to be sure we use
  # the loss across the entire tensor, we reduce it to a
  # single scalar with the mean function.
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

callbacks = [callback_early_stopping, callback_checkpoint,
  callback_tensorboard, callback_reduce_lr]

print("Would you like to perform the simulation (no, for only displaying the last simulation results.) [y/N]")

if input() is "y":
  print("Performing simulation...")
  model.fit_generator(generator=generator,
    epochs=NUM_EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_data,
    callbacks=callbacks)
else:
  print("Using old simulation checkpoint data.")

try:
    model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint: %s" % error)
    sys.exit(1)

result = model.evaluate(
  x=numpy.expand_dims(in_test_scaled, axis=0),
  y=numpy.expand_dims(out_test_scaled, axis=0))

print("loss (test-set):", result)

# Do some comparison plotting.

def plot_comparison(start_idx, length=100, train=True):
  """
  Plot the predicted and true output-signals.
  :param start_idx: Start-index for the time-series.
  :param length: Sequence-length to process and plot.
  :param train: Boolean whether to use training- or test-set.
  """
  if train:
    # Use training-data.
    inp = in_train_scaled
    out_true = out_train
  else:
    # Use test-data.
    inp = in_test_scaled
    out_true = out_test
  # End-index for the sequences.
  end_idx = start_idx + length
  # Select the sequences from the given start-index and
  # of the given length.
  inp = inp[start_idx:end_idx]
  out_true = out_true[start_idx:end_idx]
  # Input-signals for the model.
  inp = numpy.expand_dims(inp, axis=0)
  # Use the model to predict the output-signals.
  out_pred = model.predict(inp)
  # The output of the model is between 0 and 1.
  # Do an inverse map to get it back to the scale
  # of the original data-set.
  out_pred_rescaled = out_scaler.inverse_transform(out_pred[0])
  # For each output-signal.
  for signal in range(0, num_target_columns):
    # Get the output-signal predicted by the model.
    signal_pred = out_pred_rescaled[:, signal]
    # Get the true output-signal from the data-set.
    signal_true = out_true[:, signal]
    # Make the plotting-canvas bigger.
    plt.figure(figsize=(15,5))
    # Plot and compare the two signals.
    plt.plot(signal_true, label='true')
    plt.plot(signal_pred, label='pred')
    # Plot grey box for warmup-period.
    p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)
    # Plot labels etc.
    plt.ylabel(target_names[signal])
    plt.legend()
    plt.show()

# Render comparison graphs.
plot_comparison(start_idx=0, length=480, train=True)
