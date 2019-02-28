# Hello. This just my testing file for learning Keras. You can try using it if
# you want to, however, you will have to merge the dataset yourself.

import matplotlib.pyplot as plt
import numpy
import sys
import tensorflow
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from utils.loadCSV import loadInterpolatedCSV, separateCSV, loadConfigCSV
from utils.params import NetowrkHyperparameterConfig
from utils.plots import savePlotToFile, plot_comparison

# Global parameters.
DATASET_CSV_FILE = "src/utils/merged_2018_from_oct.csv"
CONFIG_LOCATION = "hyperparameters.csv"
LOADED_CONFIG_FILE = None

OUTPUT_TARGET_HEADERS = ["Gothenburg temperature", "Gothenburg wind direction", "Gothenburg wind speed",
                         "Gothenburg pressure"]

NUM_TARGET_COLUMNS = len(OUTPUT_TARGET_HEADERS)


def loadConfigFile(name):
    global LOADED_CONFIG_FILE
    LOADED_CONFIG_FILE = loadConfigCSV(CONFIG_LOCATION)


def convConfigToHyperparams(conf):
    hyperparams = NetowrkHyperparameterConfig()
    hyperparams.time_shift_in_hours = conf.time_shift
    hyperparams.training_splitting = conf.training_split
    hyperparams.start_learning_rate = conf.start_learning_rate
    hyperparams.loss_factor = conf.loss_factor
    hyperparams.min_learning_rate = conf.min_learning_rate
    hyperparams.learning_patience = conf.learning_patience
    hyperparams.num_epochs = conf.num_epochs
    hyperparams.steps_per_epoch = conf.steps_per_epoch
    hyperparams.early_stop_patience = conf.early_stop_patience
    hyperparams.warmup_steps = conf.warmup_steps
    hyperparams.num_gated_reoccurring_units = conf.num_gru
    hyperparams.activation_function = conf.activation_function
    hyperparams.network_type = conf.network_type
    hyperparams.plot_output_sub_name = "%s_" % str(conf.id)
    hyperparams.read_batch_size = conf.read_batch_size
    return hyperparams


def runConfigs():
    if LOADED_CONFIG_FILE:
        cid = 0
        conf = None
        while True:
            conf = LOADED_CONFIG_FILE.getConfig(cid)
            if conf:
                hyperparams = convConfigToHyperparams(conf)
                hyperparams.show_output_after_sim = False
                performSimulation("simulate", hyperparams)
                cid = cid + 1
            else:
                print("Ended config run at cid: %d" % cid)
                break
    else:
        print("No config CSV file has been loaded.")


# Load data from CSV files.
data = loadInterpolatedCSV(DATASET_CSV_FILE)

if not data:
    sys.exit(1)


# Visualisation.
def plotColumn(header, y_axis_unit, rows, column_idx, hyperparams):
    column = []
    for row in rows:
        column.append(row[column_idx])
    plt.figure(figsize=(15, 5))
    plt.plot(column, label=header)
    plt.ylabel(y_axis_unit)
    plt.legend()
    if cmd == "view_input_plots":
        plt.show()
    else:
        savePlotToFile("input_plot_%s" % header, "./")


class SimulationData:
  def __init__(self, hyperparams):
    hyperparams.getConfigCSV()
    (smhi_in, smhi_out) = separateCSV(data, 6, NUM_TARGET_COLUMNS)
    self.smhi_in = smhi_in
    self.smhi_out = smhi_out
    self.time_shift_hours = hyperparams.time_shift_in_hours
    self.training_split = hyperparams.training_splitting
    # Perform time shift.
    smhi_out.shiftDataColumns(- self.time_shift_hours)
    smhi_in.drop(self.time_shift_hours)
    smhi_out.pop(self.time_shift_hours)
    # Transform date format
    self.in_values = smhi_in.valuesWithDate()
    self.out_values = smhi_out.valuesWithoutDate()
    self.num_in_signals = self.in_values.shape[1]
    self.num_out_signals = self.out_values.shape[1]
  def splitAndNormalize(self):
    self.num_train = int(len(self.in_values) * self.training_split)
    self.in_train = self.in_values[0:self.num_train]
    self.in_test = self.in_values[self.num_train:]
    self.out_train = self.out_values[0:self.num_train]
    self.out_test = self.out_values[self.num_train:]
    self.in_scaler = MinMaxScaler()
    self.in_train_scaled = self.in_scaler.fit_transform(self.in_train)
    self.in_test_scaled = self.in_scaler.transform(self.in_test)
    self.out_scaler = MinMaxScaler()
    self.out_train_scaled = self.out_scaler.fit_transform(self.out_train)
    self.out_test_scaled = self.out_scaler.transform(self.out_test)
  def createValidationData(self):
    self.validation_data = (
      numpy.expand_dims(self.in_test_scaled, axis=0),
      numpy.expand_dims(self.out_test_scaled, axis=0))
  def getBatchGenerator(self, hyperparams):
    batch_size = hyperparams.read_batch_size
    sequence_length = hyperparams.read_sequence_length
    while True:
      in_shape = (batch_size, sequence_length, self.num_in_signals)
      in_batch = numpy.zeros(shape = in_shape, dtype = numpy.float16)
      out_shape = (batch_size, sequence_length, self.num_out_signals)
      out_batch = numpy.zeros(shape = out_shape, dtype = numpy.float16)
      for i in range(batch_size):
        idx = numpy.random.randint(self.num_train - sequence_length)
        in_batch[i] = self.in_train_scaled[idx : idx + sequence_length]
        out_batch[i] = self.out_train_scaled[idx : idx + sequence_length]
        yield (in_batch, out_batch)
  def lossMSEWarmup(self, out_true, out_pred):
    "Calculate the Mean Squared Error"
    out_true_slice = out_true[:, self.warmup_steps:, :]
    out_pred_slice = out_pred[:, self.warmup_steps:, :]
    self.loss = tensorflow.losses.mean_squared_error(labels = out_true_slice, predictions = out_pred_slice)
    self.loss_mean = tensorflow.reduce_mean(self.loss)
    return self.loss_mean
  def setupAndPerformSimulation(self, hyperparams):
    generator = self.getBatchGenerator(hyperparams)
    in_batch, out_batch = next(generator)
    self.model = Sequential()
    # TODO: Set the network type depending on 'NETWORK_TYPE'
    if hyperparams.network_type == "GRU":
      self.model.add(
        GRU(units = hyperparams.num_gated_reoccurring_units,
          return_sequences=True, input_shape=(None, self.num_in_signals,)))
    else:
      print("Don't know what the network type '%s' is." % hyperparams.network_type)
      sys.exit(1)
    self.model.add(Dense(self.num_out_signals, activation = hyperparams.activation_function))
    self.warmup_steps = hyperparams.warmup_steps
    optimizer = RMSprop(lr = hyperparams.start_learning_rate)
    self.model.compile(loss = self.lossMSEWarmup, optimizer = optimizer)
    self.model.summary()
    self.path_checkpoint = 'weather_sim_checkpoint.keras'
    callback_checkpoint = ModelCheckpoint(filepath = self.path_checkpoint,
      monitor = 'val_loss',
      verbose = 1,
      save_weights_only = True,
      save_best_only = True)
    callback_early_stopping = EarlyStopping(monitor = 'val_loss', patience = hyperparams.early_stop_patience, verbose = 1)
    callback_tensorboard = TensorBoard(log_dir = './keras_logs/', histogram_freq = 0, write_graph = False)
    callback_reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
      factor = hyperparams.loss_factor,
      min_lr = hyperparams.min_learning_rate,
      patience = hyperparams.learning_patience,
      verbose=1)
    self.callbacks = [callback_early_stopping, callback_checkpoint, callback_tensorboard, callback_reduce_lr]
    print("Performing simulation...")
    self.model.fit_generator(generator = generator,
      epochs = hyperparams.num_epochs,
      steps_per_epoch = hyperparams.steps_per_epoch,
      validation_data = self.validation_data,
      callbacks = self.callbacks)

def printHelp():
  print("Donno what \"%s\" is. You can use 'view_input_plots', 'save_input_plots', 'run_config' or 'simulate' at the moment." % cmd)

def performSimulation(cmd, hyperparams):
  data = SimulationData(hyperparams)
  # Parse command.
  if cmd == "view_input_plots" or cmd == "save_input_plots":
    units = ["Temperature [C]", "Wind direction [deg]", "Wind speed [m/s]", "Pressure [kilo Pascal]"]
    for i in range(2, len(data.smhi_in.header)):
      plotColumn(data.smhi_in.header[i], units[(i - 2) % len(units)], data.in_values, i + 4)
  elif cmd == "simulate":
    print("Reading %d rows of data from the dataset." % hyperparams.read_sequence_length)
    data.splitAndNormalize()
    data.createValidationData()
    data.setupAndPerformSimulation(hyperparams)
    try:
        data.model.load_weights(data.path_checkpoint)
    except Exception as error:
        print("Error trying to load checkpoint: %s" % error)
        return False
    result = data.model.evaluate(
      x=numpy.expand_dims(data.in_test_scaled, axis=0),
      y=numpy.expand_dims(data.out_test_scaled, axis=0))
    print("loss (test-set):", result)
    # Do some comparison plotting.
    def plot_comparison(start_idx, length=100, train=True):
      "Plot the predicted and true output-signals."
      if train:
        inp = data.in_train_scaled
        out_true = data.out_train
      else:
        inp = data.in_test_scaled
        out_true = data.out_test
      end_idx = start_idx + length
      inp = inp[start_idx : end_idx]
      out_true = out_true[start_idx : end_idx]
      inp = numpy.expand_dims(inp, axis=0)
      out_pred = data.model.predict(inp)
      out_pred_rescaled = data.out_scaler.inverse_transform(out_pred[0])
      for signal in range(0, NUM_TARGET_COLUMNS):
        signal_pred = out_pred_rescaled[:, signal]
        signal_true = out_true[:, signal]
        plt.figure(figsize=(15,5))
        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')
        p = plt.axvspan(0, data.warmup_steps, facecolor='black', alpha=0.15)
        header = OUTPUT_TARGET_HEADERS[signal]
        plt.ylabel(header)
        plt.legend()
        if hyperparams.show_output_after_sim:
          plt.show()
        else:
          savePlotToFile("output_plot_%s%s" % (hyperparams.plot_output_sub_name, header), hyperparams.output_folder)
    # Render comparison graphs.
    plot_comparison(start_idx = 0, length = 480, train = True)
  else:
    printHelp()
    return False
  return True

while True:
  print("Input your command (or help): ")
  cmd = input()
  if cmd == "exit":
    break
  elif cmd == "help":
    printHelp()
  elif cmd == "run_config":
    loadConfigFile(CONFIG_LOCATION)
    runConfigs()
  else:
    performSimulation(cmd, NetowrkHyperparameterConfig())
