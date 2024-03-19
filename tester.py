import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load the data set
data = yf.download('TSLA', start='2010-01-01', end='2021-01-01')

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Function to create sequences for LSTM
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i+sequence_length]
        sequences.append(sequence)
    return np.array(sequences)

# Set the sequence length
sequence_length = 10

# Create sequences for training
sequences = create_sequences(scaled_data, sequence_length)

# Separate inputs and outputs
X, y = sequences[:, :-1], sequences[:, -1]

# Reshape the data for LSTM (samples, time steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))
print(X.shape, y.shape)

# Build the LSTM model
model = Sequential()
model.add(LSTM(100, input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model on 100% of the data
model.fit(X, y, epochs=50, batch_size=32)

# Make future predictions
# Assume we want to predict 'n_future' steps into the future which corresponds to 20% of our dataset
n_future = int(len(data) * 0.2)
last_sequence = scaled_data[-sequence_length:]
future_predictions = []

for _ in range(n_future):
    # Make prediction using the last sequence
    prediction = model.predict(last_sequence.reshape(1, sequence_length, 1))
    
    # Append the prediction
    future_predictions.append(prediction[0][0])
    
    # Update the last sequence for the next prediction
    last_sequence = np.append(last_sequence[1:], prediction, axis=0)

# Invert predictions to original scale
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Generate future timestamps (assuming business day frequency)
last_date = data.index[-1]
future_dates = pd.date_range(start=last_date, periods=n_future, freq='B')

# Create a DataFrame for the future predictions with the future dates as the index
df_future = pd.DataFrame(data=future_predictions.flatten(), index=future_dates, columns=['Forecast'])

# Plot the actual data along with the future predictions
data['Close'].plot(figsize=(12, 6), legend=True)
df_future['Forecast'].plot(legend=True)