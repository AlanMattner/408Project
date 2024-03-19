import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from OldWork.userPredict import predict_future_multi_step

# https://docs.python.org/3/library/tkinter.ttk.html
# https://matplotlib.org/stable/gallery/user_interfaces/embedding_in_tk_sgskip.html


def load_stock_symbols(filename):
    try:
        with open(filename, 'r') as file:
            return [line.strip() for line in file.readlines()]
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return []

def on_predict():
    selected_stock = stock_var.get()
    # Clear previous plots from the frame
    for widget in frame.winfo_children():
        widget.destroy()
    # Generate and display new plot
    fig, ax = predict_future_multi_step(selected_stock, 3000)
    canvas = FigureCanvasTkAgg(fig, master=frame)  
    canvas.draw()
    canvas.get_tk_widget().pack()

# Load stock symbols from file
stock_symbols = load_stock_symbols('stock_symbol.txt')

# Setup the Tkinter window
window = tk.Tk()
window.title("Stock Predictor GUI")

#User input for stock symbol
stock_var = tk.StringVar()
stock_var.set(stock_symbols[0])  # Set the default option
stock_label = tk.Label(window, text="Select a stock symbol:")
stock_label.pack()
stock_dropdown = tk.OptionMenu(window, stock_var, *stock_symbols)
stock_dropdown.pack()

# Button to trigger the prediction but uses enter key as well
predict_button = tk.Button(window, text="Predict", command=on_predict)
predict_button.pack()

# Frame for displaying the plot
frame = tk.Frame(window)
frame.pack(fill=tk.BOTH, expand=True)

window.mainloop()