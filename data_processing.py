import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import datetime as dt

# Libraries used in this code (numpy, pandas, yfinance, datetime, sklearn.preprocessing.MinMaxScaler)
# https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
# https://pypi.org/project/yfinance/
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html


def download_and_scale_data(stock_symbol):
    # download the data from Yahoo Finance
    try: 
        df = yf.download(stock_symbol, start=dt.datetime(2016, 1, 1), end=dt.datetime.now())
        y = df['Close'].values.reshape(-1, 1)
        print("Data Downloaded for " + stock_symbol)

        # scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(y)
        y = scaler.transform(y)
        print("len(y):", len(y))
        return df, y, scaler
    
    except Exception as e:
        print(f"Error: {e}")
        print("Data Download Failed")
        df = None
        y = None
        scaler = None
    return

def generate_sequences(y, n_lookback = 60, n_forecast = 30):

    # create X and Y which are the input and output sequences
    # sequences are generated by taking the previous n_lookback days as the input sequence and the next n_forecast days as the output sequence
    # the sequences are generated by iterating through the data and appending the sequences to the X and Y lists

    X = []
    Y = []

    for i in range(n_lookback, len(y) - n_forecast + 1):
        X.append(y[i - n_lookback: i])
        Y.append(y[i: i + n_forecast])

    X = np.array(X)
    Y = np.array(Y)
    print("X.shape:", X.shape)
    print("Y.shape:", Y.shape)
    print("Input and Output Sequences Generated")
    print("X :", len(X))
    print("Y :", len(Y))
    return X, Y
