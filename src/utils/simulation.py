import sys

from src.utils.loadCSV import separateCSV

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import tensorflow
import numpy

from src.utils.model import loadModel


class SimulationData:
    def __init__(self, hyperparams, headers=[], data=None):
        (smhi_in, smhi_out)=separateCSV(data, 6, len(headers))
        self.smhi_in=smhi_in
        self.smhi_out=smhi_out
        self.time_shift_hours=hyperparams.time_shift_in_hours
        self.training_split=hyperparams.training_splitting

        # Perform time shift.
        smhi_out.shiftDataColumns(- self.time_shift_hours)
        smhi_in.drop(self.time_shift_hours)
        smhi_out.pop(self.time_shift_hours)

        # Transform date format
        self.in_values=smhi_in.valuesWithDate()
        self.out_values=smhi_out.valuesWithoutDate()
        self.num_in_signals = self.in_values.shape[1]
        self.num_out_signals = self.out_values.shape[1]

        self.model = None

    def splitAndNormalize(self):
        self.num_train=int(len(self.in_values)*self.training_split)
        self.in_train=self.in_values[0:self.num_train]
        self.in_test=self.in_values[self.num_train:]
        self.out_train=self.out_values[0:self.num_train]
        self.out_test=self.out_values[self.num_train:]
        self.in_scaler=MinMaxScaler()
        self.in_train_scaled=self.in_scaler.fit_transform(self.in_train)
        self.in_test_scaled=self.in_scaler.transform(self.in_test)
        self.out_scaler=MinMaxScaler()
        self.out_train_scaled=self.out_scaler.fit_transform(self.out_train)
        self.out_test_scaled=self.out_scaler.transform(self.out_test)


    def createValidationData(self):
        self.validation_data=(
            numpy.expand_dims(self.in_test_scaled, axis=0),
            numpy.expand_dims(self.out_test_scaled, axis=0))


    def getBatchGenerator(self, hyperparams):
        batch_size=hyperparams.read_batch_size
        sequence_length=hyperparams.read_sequence_length
        while True:
            in_shape=(batch_size, sequence_length, self.num_in_signals)
            in_batch=numpy.zeros(shape=in_shape, dtype=numpy.float16)
            out_shape=(batch_size, sequence_length, self.num_out_signals)
            out_batch=numpy.zeros(shape=out_shape, dtype=numpy.float16)
            for i in range(batch_size):
                idx=numpy.random.randint(self.num_train-sequence_length)
                in_batch[i]=self.in_train_scaled[idx: idx+sequence_length]
                out_batch[i]=self.out_train_scaled[idx: idx+sequence_length]
                yield (in_batch, out_batch)


    def lossMSEWarmup(self, out_true, out_pred):
        "Calculate the Mean Squared Error"
        out_true_slice=out_true[:, self.warmup_steps:, :]
        out_pred_slice=out_pred[:, self.warmup_steps:, :]
        self.loss=tensorflow.losses.mean_squared_error(labels=out_true_slice, predictions=out_pred_slice)
        self.loss_mean=tensorflow.reduce_mean(self.loss)
        return self.loss_mean

    def setupAndPerformSimulation(self, hyperparams):
        generator = self.getBatchGenerator(hyperparams)
        in_batch, out_batch=next(generator)
        self.model = loadModel(hyperparams=hyperparams, model_name="tutorial_model", in_shape=self.num_in_signals, out_shape=self.num_out_signals)
        self.model.summary()

        self.path_checkpoint=hyperparams.output_folder+'weather_sim_checkpoint.keras'
        callback_checkpoint=ModelCheckpoint(filepath=self.path_checkpoint,
                                            monitor='val_loss',
                                            verbose=1,
                                            save_weights_only=True,
                                            save_best_only=True)

        callback_early_stopping=EarlyStopping(monitor='val_loss', patience=hyperparams.early_stop_patience, verbose=1)
        callback_tensorboard=TensorBoard(log_dir=hyperparams.output_folder+'keras_logs/', histogram_freq=0,
                                         write_graph=False)
        callback_reduce_lr=ReduceLROnPlateau(monitor='val_loss',
                                             factor=hyperparams.loss_factor,
                                             min_lr=hyperparams.min_learning_rate,
                                             patience=hyperparams.learning_patience,
                                             verbose=1)

        self.callbacks=[callback_early_stopping, callback_checkpoint, callback_tensorboard, callback_reduce_lr]
        print("Performing simulation...")
        self.model.fit_generator(generator=generator,
                                 epochs=hyperparams.num_epochs,
                                 steps_per_epoch=hyperparams.steps_per_epoch,
                                 validation_data=self.validation_data,
                                 callbacks=self.callbacks)