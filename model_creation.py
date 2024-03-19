from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import tensorflow as tf

# Links to the libraries used in this code (keras, tensorflow)
# https://keras.io/api/models/sequential/
# https://keras.io/api/layers/recurrent_layers/lstm/
# https://keras.io/api/layers/core_layers/dense/
# https://keras.io/api/layers/core_layers/dropout/
# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam

def build_and_compile_model(n_lookback = 60, n_forecast = 30):
    # fit the model
    # the model is a Sequential model with two LSTM layers and a Dense layer of size n_forecast which means the model outputs a prediction of the next n_forecast days.
    # the model's input shape is (n_lookback, 1) which means the input sequences are of length n_lookback and have 1 feature.
    model = Sequential([
        LSTM(units = 100, return_sequences=True, input_shape=(n_lookback, 1)),
        Dropout(0.2),
        LSTM(units = 50, return_sequences=True),
        Dropout(0.2),
        LSTM(units = 25),
        Dense(units = n_forecast)
    ])

    # the model is compiled with the mean squared error loss function and the Adam optimizer.
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model