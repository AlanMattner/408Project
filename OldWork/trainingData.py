import numpy as np
import pandas as pd
import datetime as dt
from datetime import timedelta
import matplotlib.dates as mdates
import joblib
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import yfinance as yf
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.optimizers import Adam


#Links to the libraries used in this code
#https://pypi.org/project/yfinance/
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
#https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
#https://keras.io/api/models/sequential/
#https://keras.io/api/layers/recurrent_layers/lstm/
#https://keras.io/api/layers/regularization_layers/dropout/
#https://keras.io/api/layers/core_layers/dense/
#https://keras.io/api/optimizers/adam/
#https://keras.io/api/callbacks/early_stopping/
#https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html

yf.pdr_override()

def download_stock_data(stock_symbol, start_date, end_date):

    df = yf.download(stock_symbol, start=start_date, end=end_date)
    return df

def preprocess_data(df, feature_col='Close', n_lookback=3000, split_ratio=0.8):

    # Preprocess the data by selecting a feature, scaling it, and creating sequences for training.

    data = df[[feature_col]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Generate sequences
    X, y = [], []
    for i in range(n_lookback, len(scaled_data)):
        X.append(scaled_data[i-n_lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Split data into training and test sets
    split_idx = int(len(X) * split_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, y_train, X_test, y_test, scaler

def build_and_train_model(X_train, y_train, n_lookback=3000, steps_per_prediction=50, n_epochs=50, batch_size=16, learning_rate=0.005):

    # Build and train a complex LSTM model

    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=(n_lookback, 1)),
        Dropout(0.2),
        LSTM(units=100, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=steps_per_prediction)
    ])
    
    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    # Early stopping callback to stop training if the validation loss doesn't improve
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Fit the model
    history = model.fit(
        X_train, y_train,
        epochs=n_epochs, batch_size=batch_size,
        validation_split=0.2, callbacks=[early_stopping],
        verbose=1
    )

    # Save the model and the scaler
    model.save(f'{stock_symbol}_model.h5')
    joblib.dump(scaler, f'{stock_symbol}_scaler.joblib')
    # Write to a file the model stock symbol
    with open('stock_symbol.txt', 'a') as f:
        f.write(stock_symbol + '\n')
    
    print(f"Model and Scaler for {stock_symbol} trained and saved.")

    # Plot training & validation loss values
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()
    
    return model, history

def evaluate_model(model, X_test, y_test, scaler):
    # Predict the next sequences
    predictions_scaled = model.predict(X_test)
    predictions_rescaled = scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).reshape(-1, 50)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate MAE for each sequence and average
    mae = np.mean([mean_absolute_error(y_test_rescaled[i], predictions_rescaled[i]) for i in range(len(y_test_rescaled))])
    print(f"Average Test MAE: {mae}")
    
    # Example plotting the first actual vs predicted sequence
    plt.figure(figsize=(12, 6))
    plt.plot(range(50), y_test_rescaled[0], label='Actual')
    plt.plot(range(50), predictions_rescaled[0], label='Predicted')
    plt.legend()
    plt.title('Comparison of Actual and Predicted Sequences')
    plt.show()

if __name__ == "__main__":
    stock_symbol = input("Enter stock symbol to train and save (e.g., 'AAPL'): ")
    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime.now()
    # an array of dates from start_date to end_date
    dates = pd.date_range(start_date, end_date)
    
    # Download the stock data
    df = download_stock_data(stock_symbol, start_date, end_date)
    
    # Preprocess the data
    X_train, y_train, X_test, y_test, scaler = preprocess_data(df, feature_col='Close', n_lookback=3000, split_ratio=0.8)
    
    # Build and train the model
    model, history = build_and_train_model(X_train, y_train, n_lookback=3000, steps_per_prediction=50, n_epochs=50, batch_size=32, learning_rate=0.005)

    # Evaluate the model
    evaluate_model(model, X_test, y_test, scaler)
    

#references 
#https://realpython.com/python-gui-tkinter/
#https://waliamrinal.medium.com/long-short-term-memory-lstm-and-how-to-implement-lstm-using-python-7554e4a1776d
#https://www.geeksforgeeks.org/formatting-axes-in-python-matplotlib/