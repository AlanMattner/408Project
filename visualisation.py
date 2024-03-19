import matplotlib.pyplot as plt
import pandas as pd
# Link to the library used in this code
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.show.html


def plot_results(stock_symbol, actual_df, forecasted_values, last_actual_date):
    # Generate the future dates for the forecasted values
    future_dates = pd.date_range(start=last_actual_date + pd.Timedelta(days=1), periods=len(forecasted_values), freq='B')

    # Create a DataFrame for the forecasted values with the future dates as the index
    df_forecast = pd.DataFrame(data=forecasted_values.flatten(), index=future_dates, columns=['Forecast'])

    #https://www.w3schools.com/python/matplotlib_plotting.asp

    plt.figure(figsize=(14, 7))
    plt.plot(actual_df['Close'], label='Actual Price', color='blue', linewidth=0.5)
    plt.plot(df_forecast.index, df_forecast['Forecast'], label='Forecasted Price', color='green', linewidth=0.5)
    plt.title(f'{stock_symbol} Stock Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def plot_loss(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='loss', color='blue')
    plt.plot(history.history['val_loss'], label='val_loss', color='red')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # The plot_loss function is used to plot the training and validation loss of the model.
    # This is used to check if the model is overfitting or underfitting.