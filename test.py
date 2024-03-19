import unittest
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from model_creation import build_and_compile_model
from sklearn.preprocessing import MinMaxScaler
from data_processing import generate_sequences
from model_creation import build_and_compile_model
from forecasting import forecast, best_point_of_return, calculate_percentage_change


class TestDataProcessing(unittest.TestCase):
    def test_data_scaling(self):
        # we use a fixed DataFrame for this test.
        df = pd.DataFrame({'Close': np.arange(100)})
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

        # check if first and last values are 0 and 1, respectively, after scaling.
        self.assertEqual(scaled_data[0][0], 0)
        self.assertEqual(scaled_data[-1][0], 1)

    def test_sequence_generation(self):
        # create a dummy array for this test.
        y = np.arange(150).reshape(-1, 1)
        X, Y = generate_sequences(y, n_lookback=10, n_forecast=5)

        # check if the shapes of X and Y are as expected.
        self.assertEqual(X.shape[0], 136)
        self.assertEqual(Y.shape[0], 136)

        # check if first and last values of X and Y are as expected.
        self.assertTrue(np.array_equal(X[0], np.arange(10).reshape(-1, 1)))
        self.assertTrue(np.array_equal(Y[-1], np.arange(145, 150).reshape(-1, 1)))
        
        # check if the sequence lengths match the expected lengths.
        self.assertEqual(X.shape[1], 10)
        self.assertEqual(Y.shape[1], 5)

class TestModelCreation(unittest.TestCase):

    def test_model_creation(self):
        n_lookback = 60
        n_forecast = 30
        model = build_and_compile_model(n_lookback, n_forecast)
        
        # check the model type
        self.assertIsInstance(model, Sequential, "The model should be an instance of keras.models.Sequential")
        
        # check the input shape of the first layer
        self.assertEqual(model.layers[0].input_shape, (None, n_lookback, 1), "Input shape of the first LSTM layer is incorrect")
        
        # check the number of units in the LSTM and Dense layers
        self.assertEqual(model.layers[0].units, 100, "Number of units in the first LSTM layer is incorrect")
        self.assertEqual(model.layers[2].units, 50, "Number of units in the second LSTM layer is incorrect")
        self.assertEqual(model.layers[4].units, 25, "Number of units in the third LSTM layer is incorrect")
        self.assertEqual(model.layers[5].units, n_forecast, "Number of units in the Dense layer is incorrect")
        
        # check dropout rates
        self.assertEqual(model.layers[1].rate, 0.2, "Dropout rate after the first LSTM layer is incorrect")
        self.assertEqual(model.layers[3].rate, 0.2, "Dropout rate after the second LSTM layer is incorrect")
        
        # check the compilation configurations
        self.assertEqual(model.loss, 'mean_squared_error', "Loss function is incorrect")
        self.assertIsInstance(model.optimizer, Adam, "Optimizer should be an instance of tf.keras.optimizers.Adam")
        self.assertEqual(model.optimizer.learning_rate, 0.001, "Learning rate of the optimizer is incorrect")

    def test_best_point_of_return(self):
        # simulated forecasted values
        forecasted_values = np.array([100, 105, 110, 102, 108]).reshape(-1, 1)
        last_actual_date = pd.Timestamp('2023-12-31')

        # call the function
        best_value, best_date = best_point_of_return(forecasted_values, last_actual_date)

        # assert the best value is correct
        self.assertEqual(best_value, 110, "Incorrect best point of return value")
        
        # assert the best date is correct (3 days after the last actual date in this case)
        self.assertEqual(best_date, ('2024-01-03'), "Incorrect best point of return date")
    
    def test_calculate_percentage_change(self):
        # simulated forecasted values indicating a 10% increase
        forecasted_values = np.array([100, 101, 102, 204, 110]).reshape(-1, 1)

        # call the function
        percentage_change = calculate_percentage_change(forecasted_values)

        # assert the calculated percentage change is as expected
        self.assertAlmostEqual(percentage_change, 10.00, "Incorrect percentage change calculation")


if __name__ == "__main__":
    unittest.main()
