from data_processing import download_and_scale_data, generate_sequences
from model_creation import build_and_compile_model
from model_training import fit_model
from forecasting import forecast, best_point_of_return
from visualisation import plot_results, plot_loss

if __name__ == "__main__":
    # Main function
    stock_symbol = 'AMZN'
    df, y, scaler = download_and_scale_data(stock_symbol)
    X, Y = generate_sequences(y)
    model = build_and_compile_model()
    history = fit_model(model, X, Y, stock_symbol, epochs=50, batch_size=32)
    plot_loss(history)
    last_sequence = y[-50:]
    forecasted_values = forecast(model, scaler, last_sequence)
    best_point_of_return(forecasted_values, df.index[-1])
    plot_results(stock_symbol, df, forecasted_values, df.index[-1])

#references 
#https://realpython.com/python-gui-tkinter/
#https://waliamrinal.medium.com/long-short-term-memory-lstm-and-how-to-implement-lstm-using-python-7554e4a1776d
#https://www.geeksforgeeks.org/formatting-axes-in-python-matplotlib/
#https://www.w3schools.com/python/numpy/numpy_array_reshape.asp
#https://stackoverflow.com/questions/52821996/early-stopping-in-lstm-with-python
