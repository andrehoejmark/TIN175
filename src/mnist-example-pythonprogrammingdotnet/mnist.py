#####################################
# Example code for RNN on MNIST data
# from https://pythonprogramming.net
#####################################

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, CuDNNLSTM, LSTM, CuDNNGRU

# Load MNIST hand written digits dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scale the data (which is in greyscale)
x_train = x_train/255.0
x_test = x_test/255.0

# Select recurrent model
model = Sequential()

# The layers can be played around with (The model.add)
#
# Add LSTM layer and define input data
model.add(LSTM(128, input_shape=(x_train.shape[1:]), return_sequences='true'))
model.add(Dropout(0.2))

# The tutorial uses a second LSTM layer
model.add(LSTM(128))
# Learn more about Dropout at https://keras.io/layers/core/#dropout
model.add(Dropout(0.2))

# Learn more about Dense layers at https://keras.io/layers/core/#dense
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(10, activation='softmax'))

# learn more about optimizers at https://keras.io/optimizers/
opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# Run the model
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
