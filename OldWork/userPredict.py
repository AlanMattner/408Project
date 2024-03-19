import numpy as np
import pandas as pd
from keras.models import load_model
import joblib
import datetime as dt
from OldWork.trainingData import download_stock_data
import matplotlib.pyplot as plt


def load_data_and_model(stock_symbol, n_lookback):
    # Load the trained model and scaler
    model = load_model(f'{stock_symbol}_model.h5')
    scaler = joblib.load(f'{stock_symbol}_scaler.joblib')
    
    # Assume the existence of a function to download data. Adjust as necessary.
    end_date = dt.datetime.now()
    start_date = dt.datetime(2012, 1, 1)
    df = download_stock_data(stock_symbol, start_date, end_date)  # Reuse your existing function
    
    # Preprocess the last n_lookback days
    last_n_days = df['Close'].values[-n_lookback:].reshape(-1, 1)
    last_n_days_scaled = scaler.transform(last_n_days).reshape(1, n_lookback, 1)
    
    return last_n_days_scaled, model, scaler, df

def predict_future_multi_step(stock_symbol, n_lookback):
    # Load the trained model, scaler, and last n_lookback days of data
    last_n_days_scaled, model, scaler, df = load_data_and_model(stock_symbol, n_lookback)
    print(last_n_days_scaled)
    predictions = []
    current_input = last_n_days_scaled

    next_points_scaled = model.predict(current_input).flatten()
    # DEBUGGING
    print(next_points_scaled)
    print(next_points_scaled[0].shape)
    predictions.extend(next_points_scaled)
    # Inverse scale the predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    last_date = df.index[-1]
    predictions_array = np.array(predictions).reshape(-1, 1)  # Ensure it's a 2D array with one column
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(predictions_array), freq='D')
    df_predictions = pd.DataFrame(predictions_array, index=future_dates, columns=['Predicted Price'])
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df.index, df['Close'], label='Historical Closing Price')
    ax.plot(df_predictions.index, df_predictions['Predicted Price'], color='red', label='Future Predictions')
    
    ax.set_title(f'Future Price Prediction for {stock_symbol}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    
    print(df_predictions)
    return fig, ax

if __name__ == "__main__":
    stock_symbol = input("Enter stock symbol to predict future prices (e.g., 'AAPL'): ")
    n_lookback = 3000  # This should match the lookback period used during training
    predict_future_multi_step(stock_symbol, n_lookback)
