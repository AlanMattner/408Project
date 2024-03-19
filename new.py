import datetime as dt
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# download stcocks data from yahoo finance

df = yf.download('TSLA', start=dt.datetime(2016, 1, 1), end=dt.datetime.now())
y = df['Close'].values.reshape(-1, 1)

# scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(y)
y = scaler.transform(y)

# create X and Y which are the input and output sequences

X = []
Y = []
sequence_length = 60

for i in range(sequence_length, len(y)):
    X.append(y[i - sequence_length: i])
    Y.append(y[i])

X = np.array(X)
Y = np.array(Y)
trainsize = int(X.shape[0] * 0.80)
Xtrain = X[:int(X.shape[0] * 0.80)]
Ytrain = Y[:int(Y.shape[0] * 0.80)]
Xtest = X[int(X.shape[0] * 0.80):]
Ytest = Y[int(Y.shape[0] * 0.80):]

Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1], 1)

model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(Xtrain.shape[1], 1)))
model.add(Dense (1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(Xtrain, Ytrain, epochs=10, batch_size=16)

test_predictions = model.predict(Xtest)
test_predictions = scaler.inverse_transform(test_predictions)

Ytest = Ytest.reshape(-1, 1)
Ytest = scaler.inverse_transform(Ytest)

test_score = mean_squared_error(Ytest, test_predictions)

print('Test Score: %.2f MSE' % (test_score))

# Plotting the results
import matplotlib.pyplot as plt
plt.plot(y['Date'], y['Close'], label='Actual', color='b')
plt.plot(y['Date'].iloc[trainsize + sequence_length:], test_predictions, label='Predicted', color='r')

plt.xlabel ('Date')
plt.ylabel ('Price')
plt.title('Stock Price Prediction')

# Set the major locator to show each 3 years
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(365*3))
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: (y['Date'].iloc[0] + pd.Timedelta(days=x)).strftime('%Y')))
plt.gcf().autofmt_xdate()

