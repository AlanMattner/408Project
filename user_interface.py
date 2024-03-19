from joblib import load
from data_processing import download_and_scale_data, generate_sequences
from model_creation import build_and_compile_model
from model_training import fit_model
from forecasting import forecast, best_point_of_return, calculate_percentage_change, measure_volatility, return_on_investment
from visualisation import plot_results, plot_loss
from keras.models import load_model
import tkinter as tk
from tkinter import messagebox, ttk, PhotoImage, Label
from PIL import Image, ImageTk
import os

def on_predict():
    stock_symbol = stock_entry.get().upper()
    print(stock_symbol)

    # validation on the stock symbol to esnure it is not empty, less than 5 characters and contains only alphabets

    if stock_symbol == "":
        messagebox.showerror("Error", "Please enter a stock symbol")
        return
    elif len(stock_symbol) >= 5:
        messagebox.showerror("Error", "Stock symbol should be less than 5 characters")
        return
    elif not stock_symbol.isalpha():
        messagebox.showerror("Error", "Stock symbol should contain only alphabets")
        return
    try:
        df, y, scaler = download_and_scale_data(stock_symbol)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while fetching data (Not a real stock symbol): {e}")
        return
    
    # update UI with stock symbol
    stock_symbol_var.set(f"Stock Symbol: {stock_symbol}")

    try:

        # download and scale the data
        df, y, scaler = download_and_scale_data(stock_symbol)
        X, Y = generate_sequences(y)
        model_path = f"Models/{stock_symbol}.h5"

        # check if the model exists, if not, build and compile the model and train it
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            model = build_and_compile_model()
            fit_model(model, X, Y, stock_symbol, epochs=50, batch_size=32)

        # load the model's history
        # this is used to plot the loss and accuracy of the model
        history_path = f"Models/{stock_symbol}_history.joblib"
        history = load(history_path)

        # last 60 days of the data
        last_sequence = y[-60:]
        last_actual_price = df['Close'].iloc[-1]

        # forecast, calculate the best point of return, percentage change, volatility, and return on investment
        forecasted_values = forecast(model, scaler, last_sequence)
        best_return, best_date = best_point_of_return(forecasted_values, df.index[-1])
        percentage_change = calculate_percentage_change(forecasted_values)
        volatility = measure_volatility(forecasted_values)
        investment = return_on_investment(forecasted_values, last_actual_price)

        # update UI with the best return, percentage change, and epochs
        point_of_return_var.set(f"Best Point of Return: ${best_return} on {best_date}")
        percentage_var.set(f"Percentage Change (from first forecasted value to the last): {percentage_change}%")
        volatility_var.set(f"Volatility (a measure of changes in the forecasted values (lower = good)): {volatility}")
        investment_var.set(f"Return on Investment if you invest now: {investment}%")

        # plot the results and the loss
        plot_results(stock_symbol, df, forecasted_values, df.index[-1])
        plot_loss(history)

    except Exception as e:

        # show an error message if an error occurs
        # the error message will contain the specific error that occurred
        messagebox.showerror("Error", f"An error occurred: {e}")
        return

    messagebox.showinfo("Success", "Prediction complete!")

# main UI setup
# https://www.tutorialspoint.com/python/python_gui_programming.htm
window = tk.Tk()
window.title("Stock Price Prediction")
window.configure(bg='#383e56')

# define constants for styling
# I do this to make it easier to change the styling of the UI
BACKGROUND_COLOR = '#383e56'
TEXT_COLOR = 'white'
BUTTON_COLOR = '#f0f0f0'
FONT_FAMILY = 'Arial'
FONT_SIZE = 14

# make the grid layout responsive
# https://www.tutorialspoint.com/python/tk_grid.htm
window.grid_columnconfigure(0, weight=1)
window.grid_rowconfigure(0, weight=1)

# entry for the stock symbol
tk.Label(window, text="Enter stock symbol:", bg=BACKGROUND_COLOR, fg=TEXT_COLOR, font=(FONT_FAMILY, FONT_SIZE)).grid(row=1, column=1, padx=10, pady=5)
stock_entry = tk.Entry(window, font=(FONT_FAMILY, FONT_SIZE), justify="center")
stock_entry.grid(row=2, column=1, padx=10, pady=5)

# button to trigger the prediction
predict_button = tk.Button(window, text="Predict", bg=BUTTON_COLOR, command=on_predict)
predict_button.grid(row=3, column=1, padx=5, pady=5)

# update labels dynamically
# the text of these labels will be updated after the prediction is complete
stock_symbol_var = tk.StringVar(value="")
epochs_var = tk.StringVar(value="")
point_of_return_var = tk.StringVar(value="")
percentage_var = tk.StringVar(value="")
volatility_var = tk.StringVar(value="")
investment_var = tk.StringVar(value="")

# https://www.tutorialspoint.com/how-to-dynamically-add-remove-update-labels-in-a-tkinter-window

# labels to display dynamic values
# these labels will be updated after the prediction is complete
tk.Label(window, textvariable=stock_symbol_var, bg=BACKGROUND_COLOR, fg=TEXT_COLOR, font=(FONT_FAMILY, FONT_SIZE)).grid(row=5, column=1, padx=10, pady=5, sticky="nsew")
tk.Label(window, textvariable=point_of_return_var, bg=BACKGROUND_COLOR, fg=TEXT_COLOR, font=(FONT_FAMILY, FONT_SIZE)).grid(row=6, column=1, padx=10, pady=5, sticky="nsew")
tk.Label(window, textvariable=percentage_var, bg=BACKGROUND_COLOR, fg=TEXT_COLOR, font=(FONT_FAMILY, FONT_SIZE)).grid(row=7, column=1, padx=10, pady=5, sticky="nsew")
tk.Label(window, textvariable=volatility_var, bg=BACKGROUND_COLOR, fg=TEXT_COLOR, font=(FONT_FAMILY, FONT_SIZE)).grid(row=8, column=1, padx=10, pady=5, sticky="nsew")
tk.Label(window, textvariable=investment_var, bg=BACKGROUND_COLOR, fg=TEXT_COLOR, font=(FONT_FAMILY, FONT_SIZE)).grid(row=9, column=1, padx=10, pady=5, sticky="nsew")

# disclaimer that the predictions are for educational purposes only
# this should not be considered as financial advice as the predictions may not be accurate
tk.Label(window, text="Disclaimer: This is for educational purposes only. Do not consider this as financial advice.", bg=BACKGROUND_COLOR, fg=TEXT_COLOR, font=(FONT_FAMILY, FONT_SIZE)).grid(row=10, column=1, padx=10, pady=5, sticky="nsew")

# due to the nature of the model and volatility of the stock market, the predictions may not be accurate, and may be off by a margin of 5-10%
tk.Label(window, text="Due to the nature of the model and volatility of the stock market, the predictions may not be accurate", bg=BACKGROUND_COLOR, fg=TEXT_COLOR, font=(FONT_FAMILY, FONT_SIZE)).grid(row=11, column=1, padx=10, pady=5, sticky="nsew")
tk.Label(window, text="and may be off by a margin of 5-10%", bg=BACKGROUND_COLOR, fg=TEXT_COLOR, font=(FONT_FAMILY, FONT_SIZE)).grid(row=12, column=1, padx=10, pady=5, sticky="nsew")

# display my name
# display my email address
tk.Label(window, text="© 2024 - Alan Mattner", bg=BACKGROUND_COLOR, fg=TEXT_COLOR, font=(FONT_FAMILY, FONT_SIZE)).grid(row=13, column=1, padx=10, pady=5, sticky="nsew")
tk.Label(window, text="alan.mattner.2020@uni.strath.ac.uk", bg=BACKGROUND_COLOR, fg=TEXT_COLOR, font=(FONT_FAMILY, FONT_SIZE)).grid(row=14, column=1, padx=10, pady=5, sticky="nsew")

# load icons
# "https://www.flaticon.com/free-icons/graph" Graph icons created by Freepik 'growth.png'
# "https://www.flaticon.com/free-icons/data-modelling" Data modelling icons created by Freepik 'predictive.png'
# define the icon size
icon_width = 100
icon_height = 100

icon_path = "Icons/"
icon_files = ["growth.png"]
icons = []
for icon_file in icon_files:
    # open the icon image file
    img = Image.open(icon_path + icon_file)
    # resize the image to the desired dimensions
    img = img.resize((icon_width, icon_height), Image.ANTIALIAS)
    # convert the Image object to a PhotoImage object
    photo_img = ImageTk.PhotoImage(img)
    icons.append(photo_img)

    #https://www.geeksforgeeks.org/how-to-resize-image-in-python-tkinter/

# display icons
for i, icon in enumerate(icons, start=1):
    icon_label = Label(window, image=icon, bg='#383e56')
    icon_label.grid(row=4, column=i, padx=10, pady=5, sticky="nsew")

# start the main loop
window.mainloop()