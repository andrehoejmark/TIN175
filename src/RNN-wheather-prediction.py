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
from src.utils.loadCSV import loadInterpolatedCSV, separateCSV, loadConfigCSV
from src.utils.params import NetowrkHyperparameterConfig
from src.utils.plots import savePlotToFile, plot_comparison

# Global parameters.
DATASET_CSV_FILE = "/home/racl/School/TIN175/TIN175/data/merged.csv"
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


def performSimulation(cmd, hyperparams):
    hyperparams.getConfigCSV()
    (smhi_in, smhi_out) = separateCSV(data, 6, NUM_TARGET_COLUMNS)
    time_shift_hours = hyperparams.time_shift_in_hours
    training_split = hyperparams.training_splitting
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
        print("Reading %d rows of data from the dataset." % hyperparams.read_sequence_length)
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
                    in_batch[i] = in_train_scaled[idx:idx + sequence_length]
                    out_batch[i] = out_train_scaled[idx:idx + sequence_length]
                    yield (in_batch, out_batch)

        generator = batch_generator(hyperparams.read_batch_size, hyperparams.read_sequence_length)
        in_batch, out_batch = next(generator)
        print(in_batch.shape)
        print(out_batch.shape)

        validation_data = (
            numpy.expand_dims(in_test_scaled, axis=0),
            numpy.expand_dims(out_test_scaled, axis=0))
        # Setup network.
        model = Sequential()

        # TODO: Set the network type depending on 'NETWORK_TYPE'
        if hyperparams.network_type == "GRU":
            model.add(
                GRU(units=hyperparams.num_gated_reoccurring_units,
                    return_sequences=True, input_shape=(None, num_in_signals,)))
        else:
            print("Don't know what the network type '%s' is." % hyperparams.network_type)
            return False
        model.add(Dense(num_out_signals, activation=hyperparams.activation_function))
        warmup_steps = hyperparams.warmup_steps

        def loss_mse_warmup(out_true, out_pred):
            "Calculate the Mean Squared Error"
            out_true_slice = out_true[:, warmup_steps:, :]
            out_pred_slice = out_pred[:, warmup_steps:, :]
            loss = tensorflow.losses.mean_squared_error(labels=out_true_slice, predictions=out_pred_slice)
            loss_mean = tensorflow.reduce_mean(loss)
            return loss_mean

        optimizer = RMSprop(lr=hyperparams.start_learning_rate)
        model.compile(loss=loss_mse_warmup, optimizer=optimizer)
        model.summary()
        path_checkpoint = 'weather_sim_checkpoint.keras'
        callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                              monitor='val_loss',
                                              verbose=1,
                                              save_weights_only=True,
                                              save_best_only=True)
        callback_early_stopping = EarlyStopping(monitor='val_loss', patience=hyperparams.early_stop_patience, verbose=1)
        callback_tensorboard = TensorBoard(log_dir='./keras_logs/', histogram_freq=0, write_graph=False)
        callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                               factor=hyperparams.loss_factor,
                                               min_lr=hyperparams.min_learning_rate,
                                               patience=hyperparams.learning_patience,
                                               verbose=1)
        callbacks = [callback_early_stopping, callback_checkpoint, callback_tensorboard, callback_reduce_lr]
        print("Performing simulation...")
        model.fit_generator(generator=generator,
                            epochs=hyperparams.num_epochs,
                            steps_per_epoch=hyperparams.steps_per_epoch,
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


        # Render comparison graphs.
        plot_comparison(start_idx=0, length=480, model=model, xdata=in_test, ydata=out_test,
                        out_scaler=out_scaler, headers=OUTPUT_TARGET_HEADERS, hyperparams=hyperparams)
    else:
        print(
            "Donno what \"%s\" is. You can use 'view_input_plots', 'save_input_plots', 'run_config' or 'simulate' at the moment." % cmd)
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
        performSimulation(cmd, NetowrkHyperparameterConfig())
