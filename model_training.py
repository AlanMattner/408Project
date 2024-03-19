from keras.callbacks import EarlyStopping
import tkinter as tk
import joblib

# Links to the libraries used in this code (keras)
# https://keras.io/api/callbacks/early_stopping/

def fit_model(model, X, Y, stock_symbol, epochs = 50, batch_size = 32):
    earlyStopping  = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X, Y, epochs = epochs, batch_size = batch_size, verbose=1, validation_split=0.2, callbacks=[earlyStopping])
    print("Model Fitted")
    # Early stopping is used to stop the training if the validation loss does not improve after 4 epochs.
    # This is used to prevent overfitting.

    # Save the model to a folder called "Models" within the ProjectPython folder
    model_path = f"Models/{stock_symbol}.h5"
    model.save(model_path)
    print("Model Saved")
    # Save the model's history to a joblib file
    history_path = f"Models/{stock_symbol}_history.joblib"
    joblib.dump(history, history_path)
    print("History Saved")

    #https://www.tensorflow.org/tutorials/keras/save_and_load

    return history