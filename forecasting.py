import pandas as pd
import numpy as np

def validate_inputs(forecasted_values, last_sequence):

    # This function is used to validate the inputs of the forecast function.
    # It ensures that the forecasted_values and last_sequence are numpy arrays and that they are not empty.

    if not isinstance(forecasted_values, np.ndarray) or not len(forecasted_values):
        raise ValueError("forecasted_values must be a non-empty numpy array.")
    if not isinstance(last_sequence, np.ndarray) or not len(last_sequence):
        raise ValueError("last_sequence must be a non-empty numpy array.")
    if not isinstance(last_sequence[0], np.ndarray) or not len(last_sequence[0]):
        raise ValueError("Each element of last_sequence must be a non-empty numpy array.")

def forecast(model, scaler, last_sequence):
    validate_inputs(forecasted_values=last_sequence, last_sequence=last_sequence)
    try:

        # Predict the future values
        # I reshape the last_sequence to match the input shape of the model
        # I then use the model to predict the future values and reshape the prediction to match the output shape of the model

        prediction = model.predict(last_sequence.reshape(1, -1, 1)).reshape(-1, 1)
        forecasted_values = scaler.inverse_transform(prediction)
        print(forecasted_values)
        return forecasted_values
    except Exception as e:
        raise ValueError(f"An error occurred while forecasting: {e}")

def best_point_of_return(forecasted_values, last_actual_date):
    
    # This function is used to calculate the best point of return from the forecasted values.
    # It returns the best point of return and the date of the best point of return.

    validate_inputs(forecasted_values=forecasted_values, last_sequence=forecasted_values) 
    if not isinstance(last_actual_date, pd.Timestamp):
        raise ValueError("last_actual_date must be a valid pandas Timestamp.")

    future_dates = pd.date_range(start=last_actual_date + pd.Timedelta(days=1), periods=len(forecasted_values), freq='B')
    df_forecast = pd.DataFrame(data=forecasted_values.flatten(), index=future_dates, columns=['Forecast'])

    try:

        # I calculate the best point of return by finding the maximum forecasted value and its date

        best_point_of_return = max(df_forecast['Forecast'])
        best_point_of_return_date = df_forecast[df_forecast['Forecast'] == best_point_of_return].index[0]
        best_point_of_return = round(best_point_of_return, 2)
        best_point_of_return_date = best_point_of_return_date.strftime('%Y-%m-%d')
        print("Best Point of Return:", best_point_of_return, "Date:", best_point_of_return_date)
        return best_point_of_return, best_point_of_return_date
    except Exception as e:
        raise ValueError(f"An error occurred while calculating the best point of return: {e}")

def calculate_percentage_change(forecasted_values):
    validate_inputs(forecasted_values=forecasted_values, last_sequence=forecasted_values)
    try:

        # I calculate the percentage change by finding the difference between the last and first forecasted values
        # I then divide the difference by the first forecasted value and multiply by 100 to get the percentage change

        percentage_change = ((forecasted_values[-1] - forecasted_values[0]) / forecasted_values[0]) * 100
        percentage_change = int(percentage_change)
        print("Percentage Change:", percentage_change)
        return percentage_change
    except Exception as e:
        raise ValueError(f"An error occurred while calculating the percentage change: {e}")

def measure_volatility(forecasted_values):

    # Volatility is a measure of the dispersion of the forecasted values
    # It is calculated by finding the standard deviation of the forecasted values
    # If it is high, it means that the forecasted values are spread out over a large range of values
    # If it is low, it means that the forecasted values are concentrated around the mean

    validate_inputs(forecasted_values=forecasted_values, last_sequence=forecasted_values)
    try:
        volatility = forecasted_values.std()
        volatility = int(volatility)
        print("Volatility:", round(volatility, 2))
        return volatility
    except Exception as e:
        raise ValueError(f"An error occurred while measuring volatility: {e}")

def return_on_investment(forecasted_values, last_actual_price):

    # ROI is calculated by finding the difference between the last forecasted value and the last actual price
    # This provides a measure for the user to determine the return on investment

    validate_inputs(forecasted_values=forecasted_values, last_sequence=forecasted_values)
    if not isinstance(last_actual_price, (int, float)):
        raise ValueError("last_actual_price must be an integer or float.")
    
    try:
        roi = ((forecasted_values[-1] - last_actual_price) / last_actual_price) * 100
        roi = int(roi)
        print("Return on Investment:", roi)
        return roi
    except Exception as e:
        raise ValueError(f"An error occurred while calculating return on investment: {e}")
